import torch
import torch.nn as nn
from csqa_config import get_config
from transformers import BertModel, BertConfig
from transformers import RobertaModel, RobertaConfig
from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch.utils.data import DataLoader, RandomSampler
# from csqa_dataset import CommonsenseQA
from commonsense_dataset import CommonsenseQA
from winograde_dataset import WinoGrande
import torch.optim as optim
import torch.nn.functional as F
import time, logz, os, shutil
import os.path as osp
from tqdm import tqdm
import numpy as np
import random
import jsonlines, csv
from tensorboardX import SummaryWriter

# Dataset = {'commonsense_qa': CommonsenseQA, 'winograde': WinoGrade}
Dataset = {'commonsense_qa': CommonsenseQA, 'winogrande': WinoGrande}

Models = {
    'bert': (BertModel, BertConfig), 
    'roberta': (RobertaModel, RobertaConfig)}

class Pooler(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_size, pooler_dropout=0.0):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(pooler_dropout)

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dense(x)
        x = torch.tanh(x)
        return x

class BasicModel(nn.Module):
    def __init__(self, config):
        super(BasicModel, self).__init__()
        self.cfg = config
        for k,v in Models.items():
            if self.cfg.model.lower().startswith(k):
                text_emb_model, text_emb_config = v
                break
        text_emb_weights = osp.join(self.cfg.pretrained_lm_path, self.cfg.model.lower())
        text_emb_config = text_emb_config.from_pretrained(text_emb_weights)
        text_emb_dim = text_emb_config.hidden_size

        self.text_emb_pool = Pooler(text_emb_dim, self.cfg.pooler_dropout)
        self.text_emb_norm = nn.LayerNorm(text_emb_dim)
        # self.embedding = BertModel.from_pretrained(self.cfg.bert_path)
        
        # if self.cfg.model.lower() == 'roberta':
        #     self.linear = RobertaClassificationHead(768)
        #     self.cfg.dropout = 0
        # else:
        #     self.linear = nn.Linear(768, 1)
        self.linear = nn.Linear(text_emb_dim, 1)
        self.dropout = nn.Dropout(self.cfg.dropout)        
        self.softmax = nn.Softmax(dim=-1)
        self.apply(self._init_weights)
        self.text_embedding = text_emb_model.from_pretrained(text_emb_weights)
        if self.cfg.bert_path is not None:
            print('loading bert ckpt from ',self.cfg.bert_path)   
            assert osp.exists(self.cfg.bert_path)
            if self.cfg.cuda:
                checkpoint = torch.load(self.cfg.bert_path) 
            else:
                checkpoint = torch.load(self.cfg.bert_path, map_location=lambda storage, loc: storage)
            self.text_embedding.load_state_dict(checkpoint['net'])

    def _init_weights(self, module):
        """ Initialize the weights """
        BertLayerNorm = torch.nn.LayerNorm
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    # def forward(self, input):
    #     input_ids, input_mask, segment_ids, _, _ = input
    #     return self.step(input_ids, input_mask, segment_ids)

    def forward(self, input_ids, input_mask, segment_ids, features, fea_mask):
        _, num_choices, seq_length = input_ids.shape
        input_ids =input_ids.view(-1, seq_length)
        input_mask = input_mask.view(-1, seq_length)
        segment_ids = segment_ids.view(-1, seq_length)

        text_emb_output = self.text_embedding(input_ids,
                        attention_mask=input_mask,
                        token_type_ids=segment_ids if self.cfg.model.lower().startswith('bert') else None)

        text_emb_output = text_emb_output[0]

        output = self.text_emb_pool(text_emb_output) #sequence_output if self.cfg.model.lower() == 'roberta' else 
        if self.cfg.norm:
            output = self.text_emb_norm(output)

        if self.cfg.dropout > 0:
            output = self.dropout(output)

        logits = self.linear(output).view(-1, num_choices)

        probabilities = self.softmax(logits)
        preds = probabilities.argmax(dim=-1)

        return logits, probabilities, preds

class SelfAttention(nn.Module):
    def __init__(self, input_size, dropout=0.):
        super().__init__()
        self.input_size = input_size
        self.scorer = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.apply(self.init_weights)

    def init_weights(self, module) :
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            module.bias.data.fill_(0.1)

    def forward(self, input_seq, lens=None):
        # (B, L, H) -> (B, H)
        batch_size, seq_len, feature_dim = input_seq.size()
        input_seq = self.dropout(input_seq)
        scores = self.scorer(input_seq.contiguous().view(-1, feature_dim)).view(batch_size, seq_len)
        if lens is not None:
            for i, l in enumerate(lens):
                # print(scores.data)
                scores.data[i, l:] = -1e-15
        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(input_seq).mul(input_seq).sum(1)
        return scores, context

class FeatureModel(nn.Module):
    def __init__(self, config):
        super(FeatureModel, self).__init__()
        self.cfg = config

        for k,v in Models.items():
            if self.cfg.model.lower().startswith(k):
                text_emb_model, text_emb_config = v
                break
        text_emb_weights = osp.join(self.cfg.pretrained_lm_path, self.cfg.model.lower())
        text_emb_config = text_emb_config.from_pretrained(text_emb_weights)
        text_emb_dim = text_emb_config.hidden_size

        fea_emb_model, fea_emb_config = Models['bert']
        fea_emb_weights = osp.join(self.cfg.pretrained_lm_path, 'bert-base')
        fea_emb_config = fea_emb_config.from_pretrained(fea_emb_weights)
        fea_emb_dim = fea_emb_config.hidden_size

        self.text_emb_pool = Pooler(text_emb_dim, self.cfg.pooler_dropout)
        self.fea_emb_pool = Pooler(fea_emb_dim, self.cfg.pooler_dropout)

        if self.cfg.proj == 't2f':
            self.proj = nn.Linear(text_emb_dim, fea_emb_dim)
            self.norm = nn.LayerNorm(fea_emb_dim)
            linear_dim = fea_emb_dim * 2
        else:
            self.proj = nn.Linear(fea_emb_dim, text_emb_dim)
            self.norm = nn.LayerNorm(text_emb_dim)
            linear_dim = text_emb_dim * 2

        self.logit_fc = nn.Sequential(
            # nn.LayerNorm(self.hidden_size * 2),
            # nn.ReLU(),
            nn.Dropout (self.cfg.dropout),
            nn.Linear(linear_dim, 1)
            )

        self.softmax = nn.Softmax(dim=-1)
        self.apply(self._init_weights)
        # self.gate.apply(self.gate._init_weights)
        self.text_embedding = text_emb_model.from_pretrained(text_emb_weights)
        # if self.cfg.pretrained_bert is not None:
        #     print('loading bert ckpt from ',self.cfg.pretrained_bert)   
        #     assert osp.exists(self.cfg.pretrained_bert)
        #     if self.cfg.cuda:
        #         checkpoint = torch.load(self.cfg.pretrained_bert) 
        #     else:
        #         checkpoint = torch.load(self.cfg.pretrained_bert, map_location=lambda storage, loc: storage)
        #     self.feature.load_state_dict(checkpoint['net'])

    def _init_weights(self, module):
        """ Initialize the weights """
        BertLayerNorm = torch.nn.LayerNorm
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, (BertLayerNorm, nn.LayerNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    # def forward(self, input):
    #     input_ids, input_mask, segment_ids, features, fea_mask = input
    #     return self.step(input_ids, input_mask, segment_ids, features, fea_mask)

    def forward(self, input_ids, input_mask, segment_ids, features, fea_mask):
        bsize, num_choices, seq_length = input_ids.shape
        _, _, fea_length, hidden_size = features.shape
        input_ids =input_ids.view(-1, seq_length)
        input_mask = input_mask.view(-1, seq_length)
        fea_mask = fea_mask.view(-1, fea_length)
        segment_ids = segment_ids.view(-1, seq_length)
        features = features.view(-1, fea_length, hidden_size)
        feature, feature_weight = None, None

        extand_mask = torch.unsqueeze(input_mask, 2) 
        extand_mask = extand_mask.expand(-1,-1,hidden_size).float()

        fea_extand_mask = torch.unsqueeze(fea_mask, 2) 
        fea_extand_mask = fea_extand_mask.expand(-1,-1,hidden_size).float()

        text_emb_output = self.text_embedding(input_ids,
                        attention_mask=input_mask,
                        token_type_ids=segment_ids if self.cfg.model.lower().startswith('bert') else None)

        text_emb_output = text_emb_output[0]
        text_emb_output = self.text_emb_pool(text_emb_output)

        # feature_outputs_ = self.feature(input_ids,
        #                 attention_mask=input_mask,
        #                 token_type_ids=segment_ids)
        # sequence_output = feature_outputs_[0]
        fea_emb_output = features
        fea_emb_output = self.fea_emb_pool(fea_emb_output)

        if self.cfg.proj == 't2f':
            text_emb_output = self.proj(text_emb_output)
        else:
            fea_emb_output = self.proj(fea_emb_output)

        if self.cfg.norm:
            text_emb_output = self.norm(text_emb_output)
            fea_emb_output = self.norm(fea_emb_output)

        # if self.cfg.dropout > 0:
        #     bert_outputs = self.dropout(bert_outputs)
        #     feature_outputs = self.dropout(feature_outputs)

        feature = torch.cat([text_emb_output, fea_emb_output], dim=-1) 
        logits = self.logit_fc(feature).view(-1, num_choices)

        probabilities = self.softmax(logits)
        preds = probabilities.argmax(dim=-1)

        return logits, probabilities, preds

class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg 
        if self.cfg.feature:
            net = FeatureModel(self.cfg)
        else :
            net = BasicModel(self.cfg)
        # print(tuple(self.cfg.adam_betas))
        print(net)

        if self.cfg.cuda:
            net = net.cuda()
            if self.cfg.parallel and torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                net = nn.DataParallel(net)
        self.net = net
        self.start_epoch = 0
        if self.cfg.pretrained is not None:
            self.load_pretrained_net(pretrained = self.cfg.pretrained)

        self.index2label = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E'} if self.cfg.task == 'commonsense_qa' else {0:'1', 1:'2'}

        self.best = 1./len(self.index2label)

        if self.cfg.task == 'winograde':
            self.cfg.task += '_' + self.cfg.train_size

    def train(self, train_db, val_db):
        # if self.cfg.cuda and self.cfg.parallel:
        #     net = self.net.module
        # else:
        #     net = self.net
        # net = self.net
        # print(net)
        if self.cfg.tensorboard_logdir is not None:
            summary_writer = SummaryWriter(self.cfg.tensorboard_logdir)
        else:
            summary_writer = SummaryWriter(osp.join(self.cfg.log_dir, self.cfg.task, 'tensorboard', self.cfg.model_name))

        # log_per_steps = self.cfg.accumulation_steps * self.cfg.log_per_steps

        log_dir = osp.join(self.cfg.log_dir, self.cfg.task, self.cfg.model_name)
        if not osp.exists(log_dir):
            os.makedirs(log_dir)

        code_dir = osp.join(log_dir, 'code')
        if not osp.exists(code_dir):
            os.makedirs(code_dir)

        shutil.copy('./train.py', osp.join(code_dir, 'train.py'))
        shutil.copy('./commonsense_dataset.py', osp.join(code_dir, 'commonsense_dataset.py'))

        logz.configure_output_dir(log_dir)
        logz.save_config(self.cfg)

        train_loader = DataLoader(train_db, 
                batch_size=self.cfg.batch_size, shuffle=True, 
                num_workers=self.cfg.num_workers)

        # self.optimizer = BertAdam(net.parameters(), lr=cfg.lr, warmup=cfg.warmup)
        # self.scheduler = optim.lr_self.scheduler.StepLR(self.optimizer, step_size=3, gamma=0.8)

        num_train_steps = int(len(train_loader) / self.cfg.accumulation_steps * self.cfg.epochs)
        num_warmup_steps = int(num_train_steps * self.cfg.warmup)

        no_decay = ['bias', 'LayerNorm.weight']
        not_optim = []

        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.net.named_parameters() if (not any(nd in n for nd in no_decay))and (not any(nd in n for nd in not_optim))], 'weight_decay': self.cfg.weight_decay},
            {'params': [p for n, p in self.net.named_parameters() if (any(nd in n for nd in no_decay)) and (not any(nd in n for nd in not_optim))], 'weight_decay': 0.0}
        ]

        if self.cfg.fix_emb:
            for p in self.net.embedding.embeddings.parameters():
                p.requires_grad=False
                
        if self.cfg.ft_last_layer:
            for p in self.net.embedding.embeddings.parameters():
                p.requires_grad=False
            for i in range(10):
                for p in self.net.embedding.encoder.layer[i]:
                    p.requires_grad=False

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.cfg.lr, eps=self.cfg.adam_eps, betas=eval(self.cfg.adam_betas))
        # self.optimizer = AdamW(self.net.parameters(), lr=self.cfg.lr, eps=1e-8)


        self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps)
        loss_func = nn.CrossEntropyLoss()

        if self.cfg.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level=self.cfg.fp16_opt_level)
        # self.scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        
        # self.optimizer.set_self.scheduler(self.scheduler)

        torch.cuda.synchronize()
        self.start = time.time()
        self.net.zero_grad()
        self.batch_loss, self.batch_acc = [], []
        self.global_step = 0
        for epoch in range(self.start_epoch, self.cfg.epochs):

            print('Training...')
            torch.cuda.empty_cache()
            self.batch_loss, self.batch_acc = [], []
            for cnt, batch in tqdm(enumerate(train_loader)):
                self.net.train()
                    
                input_ids, input_mask, segment_ids, features, fea_mask, labels, input_indexs= self.batch_data(batch)
                batch_input = (input_ids, input_mask, segment_ids, features, fea_mask)
                # self.net.zero_grad()
                logits, probabilities, preds = self.net(input_ids, input_mask, segment_ids, features, fea_mask)
                loss = loss_func(logits, labels).mean()
                # print(probabilities)

                # one_hot_labels = nn.functional.one_hot(labels, num_classes = Number_class[self.cfg.task.lower()]).float()
                # per_example_loss = -torch.sum(one_hot_labels * log_probs, dim=-1)
                # loss = torch.mean(per_example_loss)

                if self.cfg.accumulation_steps > 1:
                    loss = loss/self.cfg.accumulation_steps

                if self.cfg.fp16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                    if self.cfg.max_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.cfg.max_grad_norm)
                else:
                    loss.backward()
                    if self.cfg.max_grad_norm > 0.0:
                        nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)

                acc, _, _, _ = self.evaluate(preds, labels, input_indexs)

                self.batch_loss.append(loss.cpu().data.item() / len(input_ids))
                self.batch_acc.append(acc)

                if self.global_step == 0 and cnt == 0:
                    _ = self.update_log(summary_writer, epoch, val_db)

                if ((cnt+1)%self.cfg.accumulation_steps)==0:
                    # print(nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1e5))
                    self.optimizer.step()
                    self.scheduler.step()
                    self.net.zero_grad()
                    self.global_step += 1

                    if self.global_step % self.cfg.log_per_steps == 0:
                        val_acc = self.update_log(summary_writer, epoch, val_db)
                        self.batch_loss, self.batch_acc = [], []

                        if self.cfg.save_ckpt:
                            if epoch >= (self.cfg.epochs/4):
                                if self.best < val_acc:
                                    print('Saving checkpoint...')
                                    self.save_checkpoint(epoch, acc=val_acc)
                                    self.best = val_acc

            ##################################################################
            ## Checkpoint
            ##################################################################
            if len(self.batch_loss) > 0:
                val_acc = self.update_log(summary_writer, epoch, val_db)
                self.best = max(self.best, val_acc)
                self.batch_loss, self.batch_acc = [], []

            # val_wrong_qa = []
            # for q, a in zip(val_wrong, val_wrong_answer):
            #     val_wrong_qa.append([val_db.index2qid[q], trainer.index2label[a]])
            # epoch_wrong = {epoch: val_wrong_qa}
            if self.cfg.save_ckpt:
                if epoch >= (self.cfg.epochs/4):
                    print('Saving checkpoint...')
                    self.save_checkpoint(epoch, True, acc=val_acc)
            torch.cuda.empty_cache()

        summary_writer.close()

    def update_log(self, summary_writer, epoch, val_db, inds=None):
        # print('Epoch %03d, iter %07d:'%(epoch, cnt))
        # print('loss: %05f, acc: %05f'%(np.mean(self.batch_loss), np.mean(self.batch_acc)))
        # # print(self.scheduler.get_lr()[0])
        # print('-------------------------')
        summary_writer.add_scalar('train_loss', np.mean(self.batch_loss), self.global_step)
        summary_writer.add_scalar('train_acc', np.mean(self.batch_acc), self.global_step)

        val_loss, val_acc, val_wrong, val_wrong_answer, eqs_ = self.validate(val_db)
        summary_writer.add_scalar('val_loss', np.mean(val_loss), self.global_step)
        summary_writer.add_scalar('val_acc', val_acc, self.global_step)
        summary_writer.add_scalar('lr', self.scheduler.get_lr()[0], self.global_step)

        # update optim self.scheduler
        torch.cuda.synchronize()                            
        logz.log_tabular("Time", time.time() - self.start)
        logz.log_tabular("Iteration", epoch)
        logz.log_tabular("TrainAverageLoss", np.mean(self.batch_loss))
        logz.log_tabular("TrainAverageAccu", np.mean(self.batch_acc))
        logz.log_tabular("ValAverageLoss", np.mean(val_loss))
        logz.log_tabular("ValAverageAccu", val_acc)

        if inds is not None:
            val_cnt = len(eqs_)
            eqs = [eqs_[i] for i in inds]
            eq0 = np.array(eqs[:int(val_cnt/2)])
            eq1 = np.array(eqs[int(val_cnt/2):])
            logz.log_tabular("ValAverageAccu0", eq0.sum()/len(eq0))
            logz.log_tabular("ValAverageAccu1", eq1.sum()/len(eq1))

        logz.dump_tabular()

        return val_acc


    def validate(self, val_db):
        ##################################################################
        ## Validation
        ##################################################################
        # if self.cfg.cuda and self.cfg.parallel:
        #     net = self.net.module
        # else:
        #     net = self.net
        # net = self.net
        print('Validation...')
        torch.cuda.empty_cache()
        self.net.eval()

        loss_func = nn.CrossEntropyLoss()
        

        val_loader = DataLoader(val_db, 
                batch_size=self.cfg.batch_size, shuffle=False, 
                num_workers=self.cfg.num_workers)

        val_loss, preds_, labels_, input_indexs_ = [], [], [], []
        for _, batch in tqdm(enumerate(val_loader)):
            input_ids, input_mask, segment_ids, features, fea_mask, labels, input_indexs= self.batch_data(batch)
            batch_input = (input_ids, input_mask, segment_ids, features, fea_mask)
            
            with torch.no_grad():
                logits, probabilities, preds = self.net(input_ids, input_mask, segment_ids, features, fea_mask)

                preds_.extend(preds)
                labels_.extend(labels)
                input_indexs_.extend(input_indexs)

                # if gate is not None:
                #     active_gate = torch.BoolTensor([g[0] >= 0.1 or g[1] >= 0.1 for g in gate])
                #     active_index = list(np.array(input_indexs[active_gate].cpu().data))
                #     val_activate_index += active_index
                loss = loss_func(logits, labels).mean()

                # acc, wrong_indexs, wrong_answer, eq = self.evaluate(preds, labels, input_indexs) 
                val_loss.append(loss.cpu().data.item()/len(input_ids))
                # val_acc.append(acc)
                # val_wrong += wrong_indexs
                # val_wrong_answer += wrong_answer
                # eqs.extend(eq)
          
        val_acc, val_wrong, val_wrong_answer, eqs = self.evaluate(torch.Tensor(preds_), torch.Tensor(labels_), torch.Tensor(input_indexs_)) 
        # print(val_acc)  

        return val_loss, val_acc, val_wrong, val_wrong_answer, eqs

    def test(self, test_db):
        ##################################################################
        ## Validation
        # ##################################################################
        # if self.cfg.cuda and self.cfg.parallel:
        #     net = self.net.module
        # else:
        #     net = self.net
        # net = self.net
        print('Validation...')
        torch.cuda.empty_cache()
        self.net.eval()

        test_loader = DataLoader(test_db, 
                batch_size=self.cfg.batch_size, shuffle=False, 
                num_workers=self.cfg.num_workers)

        answer = []
        
        for _, batch in tqdm(enumerate(test_loader)):
            input_ids, input_mask, segment_ids, features, fea_mask, labels, input_indexs= self.batch_data(batch)
            batch_input = (input_ids, input_mask, segment_ids, features, fea_mask)

            with torch.no_grad():
                logits, probabilities, preds = self.net(input_ids, input_mask, segment_ids, features, fea_mask)
                input_indexs = list(np.array(input_indexs.cpu().data))
                preds = list(np.array(preds.cpu().data))
                for ind, pred in zip(input_indexs, preds):
                    answer.append(( test_db.index2qid[ind], self.index2label[pred]))

        return answer

    def load_pretrained_net(self, pretrained):
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net
        # self.begin_epoch = int(pretrained_name.split('-')[1]) + 1
        print('loading ckpt from ',pretrained)
        
        assert osp.exists(pretrained)
        if self.cfg.cuda:
            checkpoint = torch.load(pretrained) 
        else:
            checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage)

        net.load_state_dict(checkpoint['net'])
        

    def save_checkpoint(self, epoch, force=False, acc=None):
        # wrong_index_path = osp.join(self.cfg.log_dir, self.cfg.task, self.cfg.model_name, "wrong_index.jsonl")
        # with jsonlines.open(wrong_index_path, 'a+') as writer:
        #     writer.write(epoch_wrong)

        print(" [*] Saving checkpoints...")
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        checkpoint_dir = osp.join(self.cfg.log_dir, self.cfg.task, self.cfg.model_name)
        if not osp.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        tail = ''
        if acc is not None:
            acc = str(round(acc, 5))[2:]
            tail += '-'
            tail += acc
        if force:
            tail += '-'
            tail += 'end'
        model_name = "ckpt-%03d%s.pkl" % (epoch, tail)

        print('saving ckpt to ', checkpoint_dir)
        if self.cfg.fp16:
            state = {'net':net.state_dict(), 'optimizer':self.optimizer.state_dict(), 'epoch':epoch, 'amp': amp.state_dict()}
        else:
            state = {'net':net.state_dict(), 'optimizer':self.optimizer.state_dict(), 'epoch':epoch}
        # torch.save(net.state_dict(), osp.join(checkpoint_dir, model_name))
        torch.save(state, osp.join(checkpoint_dir, model_name))

    def batch_data(self, entry):
        features, fea_mask = None, None
        input_ids = entry['token_ids'].long()
        segment_ids = entry['segment_ids'].long()
        input_mask = entry['mask'].long()
        labels = entry['label_ids'].long()
        input_indexs = entry['index'].long()
        
        if self.cfg.feature:
            features = entry['feature'].float()
            fea_mask = entry['fea_mask'].long()

        # print(input_ids[0])
        # exit()

        if self.cfg.cuda:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda() 
            labels = labels.cuda()
            input_indexs = input_indexs.cuda()
            if self.cfg.feature:
                features = features.cuda()
                fea_mask = fea_mask.cuda()
        
        return input_ids, input_mask, segment_ids, features, fea_mask, labels, input_indexs

    def evaluate(self, pred, labels, input_indexs):
        eq = torch.eq(pred, labels)
        # print(labels.shape)
        wrong_indexs = list(np.array(input_indexs[~eq].cpu().data))
        wrong_answer = list(np.array(pred[~eq].cpu().data))
        correct = eq.sum().cpu().data.item()
        acc = correct/len(labels)

        return acc, wrong_indexs, wrong_answer, np.array(eq.cpu())



if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)

    cfg, unparsed = get_config()

    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if(cfg.cuda):
        torch.cuda.manual_seed_all(cfg.seed)

    if not cfg.do_train and not cfg.do_eval and not cfg.do_pred:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")
    
    # bert = BertModel.from_pretrained(UNCASED, output_hidden_states=True)
    # bert_config = BertConfig.from_json_file(FLAGS.bert_config_file)

    # if cfg.max_seq_length > bert_config.max_position_embeddings:
    #     raise ValueError(
    #         "Cannot use sequence length %d because the BERT model "
    #         "was only trained up to sequence length %d" %
    #         (cfg.max_seq_length, bert_config.max_position_embeddings))

    # tf.gfile.MakeDirs(FLAGS.output_dir)

    # processor = CommonsenseQAProcessor(split=FLAGS.split)

    if cfg.task is None:
        raise ValueError(
            "Task needs to be specified. commonsense_qa / winograde")

    dataset = Dataset[cfg.task.lower()]    
    trainer = Trainer(cfg)

    if cfg.do_train:
        train_db = dataset(cfg, split='train')
        val_db   = dataset(cfg, split='val')
        trainer.train(train_db, val_db)

    if cfg.do_eval:
        val_db   = dataset(cfg, split='val')
        

        val_loss, val_acc, val_wrong, val_wrong_answer, eqs = trainer.validate(val_db)
        print("ValAverageLoss", np.mean(val_loss))
        print("ValAverageAccu", val_acc)

        torch.cuda.empty_cache()

    if cfg.do_pred:
        print('preprocessing data...')
        test_db   = dataset(cfg, split='test')
        print('predicting...')
        answer = trainer.test(test_db)

        test_pred_path = osp.join(cfg.log_dir, cfg.task, cfg.prediction_file)
        with open(test_pred_path,'w')as f:
            f_csv = csv.writer(f)
            f_csv.writerows(answer)
        # with jsonlines.open(test_pred_path, 'a') as writer:
        #     for a in answer:
        #         writer.write(a)