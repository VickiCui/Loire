#!/usr/bin/env python
# codes modified from
#   https://github.com/uvavision/Text2Scene/blob/master/lib/modules/layout_trainer.py

import os, sys, cv2, math, gc
from tqdm import tqdm
import random, json, logz
import numpy as np
import os.path as osp
from copy import deepcopy
import matplotlib.pyplot as plt
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from layout_utils import *
from layout_config import get_config
from modules.evaluator import *
from modules.my_model import DrawModel
from optim import Optimizer
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import Sampler
from random import shuffle
from transformers import AdamW, get_linear_schedule_with_warmup

class SupervisedTrainer(object):
    def __init__(self, db):
        self.cfg = db.cfg 
        self.db = db
        net = DrawModel(db, if_bert=True)

        if self.cfg.cuda:
            if self.cfg.parallel and torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                net = nn.DataParallel(net)
            net = net.cuda()
        self.net = net
        self.start_epoch = 0
        if self.cfg.pretrained is not None:
            self.load_pretrained_net(self.cfg.pretrained)
        self.writer = SummaryWriter(comment='Layout')
        self.xymap = CocoLocationMap(self.cfg)
        self.whmap = CocoTransformationMap(self.cfg)
        self.optimizer = None
        self.bert_optimizer = None
        self.bert_scheduler = None
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.train_bucket_sampler = None
        self.val_bucket_sampler = None
        self.global_step = 0

    def get_parameter_number(self):
        net = self.net
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('Total', total_num, 'Trainable', trainable_num)
        
    # def load_optimizer(self, pretrained_name):
    #     cache_dir = osp.join(self.cfg.data_dir, 'caches')
    #     pretrained_path = osp.join(cache_dir, 'bert_layout_ckpts', pretrained_name+'.pkl')
    #     bert_path = osp.join(cache_dir, 'bert_layout_ckpts', 'bert-'+pretrained_name+'.pkl')
        
    #     assert osp.exists(pretrained_path)
    #     if self.cfg.cuda:
    #         checkpoint = torch.load(pretrained_path) 
    #         bert_checkpoint = torch.load(bert_path) 
    #     else:
    #         checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage) 
    #         bert_checkpoint = torch.load(bert_path, map_location=lambda storage, loc: storage) 
    #     self.optimizer.optimizer.load_state_dict(checkpoint['optimizer'])
    #     self.bert_optimizer.load_state_dict(bert_checkpoint['optimizer'])
        
    def load_pretrained_net(self, pretrained_name):
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        # cache_dir = osp.join(self.cfg.data_dir, 'caches')
        # pretrained_path = osp.join(cache_dir, 'bert_layout_ckpts', pretrained_name+'.pkl')
        pretrained_path = pretrained_name
        # self.begin_epoch = int(pretrained_name.split('-')[1]) + 1
        print('loading ckpt from ',pretrained_path)
        
        assert osp.exists(pretrained_path)
        if self.cfg.cuda:
            checkpoint = torch.load(pretrained_path) 
        else:
            checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage) 
        net.load_state_dict(checkpoint['net'])
        # self.optimizer.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch'] + 1
        print('Start training from {} epoch'.format(self.start_epoch))

    def batch_data(self, entry):
        ################################################
        # Inputs
        maxlen = max(entry['obj_cnt']).item()
        # print(entry['obj_cnt'])
        input_inds = entry['bert_inds'].long()
        input_lens = entry['bert_lens'].unsqueeze(-1).long()
        fg_inds = entry['fg_inds'].long()
        bg_imgs = entry['background'].float()
        fg_onehots = indices2onehots(fg_inds, self.cfg.output_cls_size)

        ################################################
        # Outputs
        ################################################
        gt_inds = entry['out_inds'].long()
        gt_msks = entry['out_msks'].float()
        gt_scene_inds = entry['scene_idx'].long().numpy()
        gt_boxes = entry['boxes'].float()
        box_msks = entry['box_msks'].float()

        ################################################
        # print(entry['obj_cnt'][0])
        # print(fg_onehots[0])
        # print(gt_inds[0])


        if self.cfg.cuda:
            input_inds = input_inds.cuda()
            input_lens = input_lens.cuda()
            fg_onehots = fg_onehots[:, :maxlen+1, :].cuda() #(bsize,11,83)
            bg_imgs = bg_imgs[:, :maxlen, :, :, :].cuda() #(bsize, 10, 83, 64, 64)
            gt_inds = gt_inds[:, :maxlen, :].cuda() #(bsize, 10,4)
            gt_msks = gt_msks[:, :maxlen, :].cuda() #(bsize, 10,4)
            gt_boxes = gt_boxes[:, :maxlen, :].cuda() #(bsize, 10,5)
            box_msks = box_msks[:, :maxlen, :].cuda() #(bsize, 10,5)
        # print(input_inds.shape, input_lens.shape, fg_onehots.shape, bg_imgs.shape, 
        # gt_inds.shape, gt_msks.shape, gt_boxes.shape, box_msks.shape)
        
        return input_inds, input_lens, bg_imgs, fg_onehots, gt_inds, gt_msks, gt_scene_inds, gt_boxes, box_msks
    
    def evaluate(self, inf_outs, ref_inds, ref_msks, ref_boxes, box_msks):
        ####################################################################
        # Prediction loss
        ####################################################################
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net
        if not self.cfg.mse:
            _, _, _, enc_msks, what_wei, where_wei = inf_outs
        else:
            obj_logits, pred_boxes, enc_msks, what_wei, where_wei = inf_outs

        ####################################################################
        # doubly stochastic attn loss
        ####################################################################
        attn_loss = 0
        encoder_msks = enc_msks

        if self.cfg.what_attn:
            obj_msks = ref_msks[:,:,0].unsqueeze(-1)
            what_att_logits = what_wei
            raw_obj_att_loss = torch.mul(what_att_logits, obj_msks)
            raw_obj_att_loss = torch.sum(raw_obj_att_loss, dim=1)
            obj_att_loss = raw_obj_att_loss - encoder_msks
            obj_att_loss = torch.sum(obj_att_loss ** 2, dim=-1)
            obj_att_loss = torch.mean(obj_att_loss)
            attn_loss = attn_loss + obj_att_loss

        attn_loss = self.cfg.attn_loss_weight * attn_loss

        eos_loss = 0
        if self.cfg.what_attn and self.cfg.eos_loss_weight > 0:
            # print('-------------------')
            # print('obj_msks: ', obj_msks.size())
            inds_1 = torch.sum(obj_msks, 1, keepdim=True) - 1
            # print('inds_1: ', inds_1.size())
            bsize, tlen, slen = what_att_logits.size()
            # print('what_att_logits: ', what_att_logits.size())
            inds_1 = inds_1.expand(bsize, 1, slen).long()
            local_eos_probs = torch.gather(what_att_logits, 1, inds_1).squeeze(1)
            # print('local_eos_probs: ', local_eos_probs.size())
            # print('encoder_msks: ', encoder_msks.size())
            inds_2 = torch.sum(encoder_msks, 1, keepdim=True) - 1
            # print('inds_2: ', inds_2.size())
            eos_probs  = torch.gather(local_eos_probs, 1, inds_2.long())
            norm_probs = torch.gather(raw_obj_att_loss, 1, inds_2.long())
            # print('norm_probs:', norm_probs.size())
            # print('eos_probs: ', eos_probs.size())
            eos_loss = -torch.log(eos_probs.clamp(min=self.cfg.eps))
            eos_loss = torch.mean(eos_loss)
            diff = torch.sum(norm_probs) - 1.0
            norm_loss = diff * diff
            # print('obj_att_loss: ', att_loss)
            # print('eos_loss: ', eos_loss)
            # print('norm_loss: ', norm_loss)
        eos_loss = self.cfg.eos_loss_weight * eos_loss
        
        # torch.cuda.synchronize()
        # s = time()

        if not self.cfg.mse:
            # _, _, _, enc_msks, what_wei, where_wei = inf_outs

            logits = net.collect_logits(inf_outs, ref_inds)
            bsize, slen, _ = logits.size()
            loss_wei = [
                self.cfg.obj_loss_weight, \
                self.cfg.coord_loss_weight, \
                self.cfg.scale_loss_weight, \
                self.cfg.ratio_loss_weight
            ]
            loss_wei = torch.from_numpy(np.array(loss_wei)).float()
            if self.cfg.cuda:
                loss_wei = loss_wei.cuda()
            loss_wei = loss_wei.view(1,1,4)
            loss_wei = loss_wei.expand(bsize, slen, 4)

            pred_loss = -torch.log(logits.clamp(min=self.cfg.eps)) * loss_wei * ref_msks
            pred_loss = torch.sum(pred_loss)/(torch.sum(ref_msks) + self.cfg.eps)

            ####################################################################
            # Accuracies
            ####################################################################
            pred_accu, pred_mse = net.collect_accuracies(inf_outs, ref_inds)
            pred_accu = pred_accu * ref_msks

            mse_msks = ref_msks[:,:,-1].unsqueeze(-1).expand(-1,-1,4)
            pred_mse = pred_mse * mse_msks

            comp_accu = torch.sum(torch.sum(pred_accu, 0), 0)
            comp_msks = torch.sum(torch.sum(ref_msks, 0), 0)
            pred_accu = comp_accu/(comp_msks + self.cfg.eps)

            comp_mse = torch.sum(torch.sum(pred_mse, 0), 0)
            comp_msks = torch.sum(torch.sum(mse_msks, 0), 0)
            pred_mse = comp_mse/(comp_msks + self.cfg.eps)
        
        else:
            #inf_outs = (obj_logits, pred_boxes, enc_msks, what_wei, where_wei)
            # obj_logits, pred_boxes, enc_msks, what_wei, where_wei = inf_outs
            b_size, tlen, _ = pred_boxes.size()
            ref_boxes = ref_boxes * box_msks
            
            # logits = net.collect_logits(inf_outs, ref_inds)
            obj_inds   = ref_inds[:, :, 0].unsqueeze(-1)
            sample_obj_logits   = torch.gather(obj_logits, -1, obj_inds)

            obj_loss = -torch.log(sample_obj_logits.clamp(min=self.cfg.eps))  * ref_msks[:,:,0].unsqueeze(-1)
            obj_loss = torch.sum(obj_loss)/(torch.sum(ref_msks[:,:,0]) + self.cfg.eps)

            # obj_logits = obj_logits.float() * box_msks[:,:,0].unsqueeze(-1).expand(-1,-1,83)
            pred_boxes = pred_boxes * box_msks[:,:,1:]
            
            gt_obj   = ref_boxes[:, :, 0].unsqueeze(-1).long()
            gt_boxes = ref_boxes[:, :, 1:]

            # loss_wei = [
            #     self.cfg.obj_loss_weight, \
            #     self.cfg.coord_loss_weight, \
            #     self.cfg.scale_loss_weight, \
            #     self.cfg.ratio_loss_weight
            # ]
            # loss_wei = torch.from_numpy(np.array(loss_wei)).float()
            # if self.cfg.cuda:
            #     loss_wei = loss_wei.cuda()
            # loss_wei = loss_wei.view(1,1,4)
            # loss_wei = loss_wei.expand(bsize, slen, 4)

            # torch.cuda.synchronize()
            # print(time()-s)
            # torch.cuda.synchronize()
            # s = time()

            all_mse = (pred_boxes - gt_boxes)**2

            comp_mse = torch.sum(torch.sum(all_mse, 0), 0)
            comp_msks = torch.sum(torch.sum(box_msks[:,:,1:], 0), 0)
            pred_mse = comp_mse/(comp_msks + self.cfg.eps)
            coord_loss = torch.sum(pred_mse)

            pred_loss = obj_loss + coord_loss
            # pred_loss = obj_loss + x_loss + y_loss + w_loss + h_loss
            # torch.cuda.synchronize()
            # print(time()-s)
            # torch.cuda.synchronize()
            # s = time()

            ####################################################################
            # Accuracies
            ####################################################################
            coord = self.db.boxes2indices(pred_boxes.contiguous().view(-1,4).detach().cpu().numpy())
            coord = torch.from_numpy(coord).view(b_size, tlen, -1).cuda()

            _, pred_obj_inds   = torch.max(obj_logits,   -1)
            obj_accu   = torch.eq(pred_obj_inds, ref_inds[:, :, 0]).float().unsqueeze(-1)
            coord_accu = torch.eq(coord, ref_inds[:,:,1:]).float()

            # pred_accu, pred_mse = net.collect_accuracies(inf_outs, ref_inds)
            pred_accu = torch.cat([obj_accu, coord_accu], -1)
            pred_accu = pred_accu * ref_msks

            comp_accu = torch.sum(torch.sum(pred_accu, 0), 0)
            comp_msks = torch.sum(torch.sum(ref_msks, 0), 0)
            pred_accu = comp_accu/(comp_msks + self.cfg.eps)

            # pred_mse = torch.stack([x_loss, y_loss, w_loss, h_loss], -1)
            # torch.cuda.synchronize()
            # print(time()-s)
            # print('====================')

        return pred_loss, attn_loss, eos_loss, pred_accu, pred_mse
        
    def train(self, train_db, val_db, test_db):
        ##################################################################
        ## Optimizer
        ##################################################################
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net
        image_encoder_trainable_paras = \
            filter(lambda p: p.requires_grad, net.image_encoder.parameters())
        # raw_optimizer = optim.Adam([
        #         {'params': net.text_encoder.parameters(), 'lr': self.cfg.finetune_lr},
        #         {'params': image_encoder_trainable_paras},
        #         {'params': net.what_decoder.parameters()}, 
        #         {'params': net.where_decoder.parameters()}
        #     ], lr=self.cfg.lr)
        raw_optimizer = optim.Adam([
                {'params': image_encoder_trainable_paras, 'initial_lr': self.cfg.lr},
                {'params': net.what_decoder.parameters(), 'initial_lr': self.cfg.lr}, 
                {'params': net.where_decoder.parameters(), 'initial_lr': self.cfg.lr}
            ], lr=self.cfg.lr)
        self.optimizer = Optimizer(raw_optimizer, max_grad_norm=self.cfg.grad_norm_clipping)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer.optimizer, factor=0.8, patience=3)
        scheduler = optim.lr_scheduler.StepLR(self.optimizer.optimizer, step_size=3, gamma=0.8, last_epoch = self.start_epoch-1)
        self.optimizer.set_scheduler(scheduler)

        num_train_steps = int(len(train_db) / self.cfg.accumulation_steps * self.cfg.n_epochs)
        num_warmup_steps = int(num_train_steps * self.cfg.warmup)
        self.bert_optimizer = AdamW([{'params': net.text_encoder.parameters(), 'initial_lr': self.cfg.finetune_lr}], lr=self.cfg.finetune_lr)
        self.bert_scheduler = get_linear_schedule_with_warmup(self.bert_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps, last_epoch = self.start_epoch-1)

        bucket_boundaries = [4,8,12,16,22]  # [4,8,12,16,22]
        print('preparing training bucket sampler')
        self.train_bucket_sampler = BucketSampler(train_db, bucket_boundaries, batch_size=self.cfg.batch_size)
        print('preparing validation bucket sampler')
        self.val_bucket_sampler = BucketSampler(val_db, bucket_boundaries, batch_size=4)

        ##################################################################
        ## LOG
        ##################################################################
        logz.configure_output_dir(self.cfg.model_dir)
        logz.save_config(self.cfg)

        ##################################################################
        ## Main loop
        ##################################################################
        start = time()
        
        for epoch in range(self.start_epoch, self.cfg.n_epochs):
            ##################################################################
            ## Training
            ##################################################################
            print('Training...')
            torch.cuda.empty_cache()
            train_pred_loss, train_attn_loss, train_eos_loss, train_accu, train_mse = \
                self.train_epoch(train_db, self.optimizer, epoch)
        
            ##################################################################
            ## Validation
            ##################################################################
            print('Validation...')
            val_loss, val_accu, val_mse, val_infos = self.validate_epoch(val_db)
            
            ##################################################################
            ## Sample
            ##################################################################
            if self.cfg.if_sample:
                print('Sample...')
                torch.cuda.empty_cache()
                self.sample(epoch, test_db, self.cfg.n_samples)
                torch.cuda.empty_cache()
            ##################################################################
            ## Logging
            ##################################################################

            # update optim scheduler
            print('Loging...')
            self.optimizer.update(np.mean(val_loss), epoch)
                
            logz.log_tabular("Time", time() - start)
            logz.log_tabular("Iteration", epoch)

            logz.log_tabular("TrainAverageError", np.mean(train_pred_loss))
            logz.log_tabular("TrainAverageAccu", np.mean(train_accu))
            logz.log_tabular("TrainAverageMse", np.mean(train_mse))
            logz.log_tabular("ValAverageError", np.mean(val_loss))
            logz.log_tabular("ValAverageAccu", np.mean(val_accu))
            logz.log_tabular("ValAverageObjAccu", np.mean(val_accu[:, 0]))
            logz.log_tabular("ValAverageCoordAccu", np.mean(val_accu[:, 1]))
            logz.log_tabular("ValAverageScaleAccu", np.mean(val_accu[:, 2]))
            logz.log_tabular("ValAverageRatioAccu", np.mean(val_accu[:, 3]))
            logz.log_tabular("ValAverageMse", np.mean(val_mse))
            logz.log_tabular("ValAverageXMse", np.mean(val_mse[:, 0]))
            logz.log_tabular("ValAverageYMse", np.mean(val_mse[:, 1]))
            logz.log_tabular("ValAverageWMse", np.mean(val_mse[:, 2]))
            logz.log_tabular("ValAverageHMse", np.mean(val_mse[:, 3]))
            logz.log_tabular("ValUnigramF3", np.mean(val_infos.unigram_F3()))
            logz.log_tabular("ValBigramF3",  np.mean(val_infos.bigram_F3()))
            logz.log_tabular("ValUnigramP",  np.mean(val_infos.unigram_P()))
            logz.log_tabular("ValUnigramR",  np.mean(val_infos.unigram_R()))
            logz.log_tabular("ValBigramP",   val_infos.mean_bigram_P())
            logz.log_tabular("ValBigramR",   val_infos.mean_bigram_R())
            logz.log_tabular("ValUnigramScale", np.mean(val_infos.scale()))
            logz.log_tabular("ValUnigramRatio", np.mean(val_infos.ratio()))
            logz.log_tabular("ValUnigramSim",   np.mean(val_infos.unigram_coord()))
            logz.log_tabular("ValBigramSim",    val_infos.mean_bigram_coord())

            logz.dump_tabular()

            ##################################################################
            ## Checkpoint
            ##################################################################
            print('Saving checkpoint...')
            log_info = [np.mean(val_loss), np.mean(val_accu)]
            self.save_checkpoint(epoch, log_info)
            torch.cuda.empty_cache()

    def train_epoch(self, train_db, optimizer, epoch):
        train_db.cfg.sent_group = -1

        train_loader = DataLoader(train_db, 
            batch_size=1,
            num_workers=self.cfg.num_workers,
            batch_sampler=self.train_bucket_sampler,
            drop_last=False,
            pin_memory = False)

        train_pred_loss, train_attn_loss, train_eos_loss, train_accu, train_mse = [], [], [], [], []

        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        get_data = 0.
        forward = 0.
        eva = 0.
        back = 0.
        opt = 0.
        start = time()
        for cnt, batched in tqdm(enumerate(train_loader)):
            ##################################################################
            ## Batched data
            #############################F#####################################
            # torch.cuda.synchronize()
            # s = time()
            input_sentences, input_lens, bg_imgs, fg_onehots, \
            gt_inds, gt_msks, gt_scene_inds, boxes, box_msks = \
                self.batch_data(batched)
            gt_scenes = [deepcopy(train_db.scenedb[x]) for x in gt_scene_inds]
                
            ##################################################################
            ## Train one step
            ##################################################################
            self.net.train()
            # torch.cuda.synchronize()
            # get_data += time()-s
            # torch.cuda.synchronize()
            # s = time()


            if self.cfg.teacher_forcing:
                # self.get_parameter_number()
                # net.get_parameter_number()
                inputs = (input_sentences, input_lens, bg_imgs, fg_onehots)
                # inputs = (torch.zeros(8, 25).long().cuda(), torch.zeros(8).long().cuda(), torch.zeros(8, 10, 83, 64, 64).float().cuda(), torch.zeros(8,11,83).float().cuda())
                # self.writer.add_graph(self.net, inputs, True)
                # sys.exit()
                inf_outs, _ = self.net(inputs)
            else:
                inf_outs, _ = net.inference(input_sentences, input_lens, -1, -0.1, 0, gt_inds)

            # torch.cuda.synchronize()
            # forward += time()-s
            # torch.cuda.synchronize()
            # s = time()

            pred_loss, attn_loss, eos_loss, pred_accu, pred_mse = self.evaluate(inf_outs, gt_inds, gt_msks, boxes, box_msks)
            # torch.cuda.synchronize()
            # eva += time()-s
            # torch.cuda.synchronize()
            # s = time()
            

            loss = pred_loss + attn_loss + eos_loss
            loss = loss/self.cfg.accumulation_steps

            loss.backward()
            # torch.cuda.synchronize()
            # back += time()-s
            # torch.cuda.synchronize()
            # s = time()

            if((cnt+1)%self.cfg.accumulation_steps)==0:
                self.optimizer.step()
                self.bert_optimizer.step()
                self.bert_scheduler.step()
                self.net.zero_grad()
                self.global_step += 1

                # torch.cuda.synchronize()
                # opt += time()-s

                ##################################################################
                ## Collect info
                ##################################################################
                train_pred_loss.append(pred_loss.cpu().data.item())
                if attn_loss == 0:
                    attn_loss_np = 0
                else:
                    attn_loss_np = attn_loss.cpu().data.item()
                train_attn_loss.append(attn_loss_np)
                if eos_loss == 0:
                    eos_loss_np = 0
                else:
                    eos_loss_np = eos_loss.cpu().data.item()
                train_eos_loss.append(eos_loss_np)
                train_accu.append(pred_accu.cpu().data.numpy())
                train_mse.append(pred_mse.cpu().data.numpy())


                ##################################################################
                ## Print info
                ##################################################################
                if self.global_step % self.cfg.log_per_steps == 0:
                    print('Epoch %03d, iter %07d:'%(epoch, cnt))
                    print('loss: ', np.mean(train_pred_loss), np.mean(train_attn_loss), np.mean(train_eos_loss))
                    print('accu: ', np.mean(np.array(train_accu), 0))
                    print('xy & wh mse: ', np.mean(np.array(train_mse), 0))
                    torch.cuda.synchronize()
                    print('-------------------------')

        return train_pred_loss, train_attn_loss, train_eos_loss, train_accu, train_mse

    def validate_epoch(self, val_db):
        val_loss, val_accu, val_mse, top1_scores = [], [], [], []
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        for G in range(1):
            val_db.cfg.sent_group = G

            val_loader = DataLoader(val_db, 
                        batch_size=1,
                        num_workers=self.cfg.num_workers,
                        batch_sampler=self.val_bucket_sampler,
                        drop_last=False,
                        pin_memory = False)
            
            for cnt, batched in tqdm(enumerate(val_loader)):
                ##################################################################
                ## Batched data
                ##################################################################
                input_inds, input_lens, bg_imgs, fg_onehots, \
                gt_inds, gt_msks, gt_scene_inds, boxes, box_msks = \
                    self.batch_data(batched)
                gt_scenes = [deepcopy(val_db.scenedb[x]) for x in gt_scene_inds]

                ##################################################################
                ## Validate one step
                ##################################################################
                self.net.eval()
                with torch.no_grad():
                    _, env = net.inference(input_inds, input_lens, -1, 2.0, 0, None)
                    # infos = env.batch_evaluation(gt_inds.cpu().data.numpy())
                    scores = env.batch_evaluation(gt_scenes)
                    # scores = np.stack(scores, 0)
                    # infos = eval_info(self.cfg, scores)
                    inputs = (input_inds, input_lens, bg_imgs, fg_onehots)
                    inf_outs, _ = self.net(inputs)
                    # inf_outs, _ = self.net.teacher_forcing(input_inds, input_lens, bg_imgs, fg_onehots)
                    # print('gt_inds', gt_inds)
                    pred_loss, attn_loss, eos_loss, pred_accu, pred_mse = self.evaluate(inf_outs, gt_inds, gt_msks, boxes, box_msks) #self.evaluate(inf_outs, gt_inds, gt_msks)
                
                top1_scores.extend(scores)
                val_loss.append(pred_loss.cpu().data.item())
                val_accu.append(pred_accu.cpu().data.numpy())  
                val_mse.append(pred_mse.cpu().data.numpy())  

                # print(G, cnt)
                # print('pred_loss', pred_loss.data.item())
                # print('pred_accu', pred_accu)
                # print('scores', scores)
                # if cnt > 0:
                #     break
        
        top1_scores = np.stack(top1_scores, 0)
        val_loss = np.array(val_loss)
        val_accu = np.stack(val_accu, 0)
        val_mse = np.stack(val_mse, 0)
        infos = eval_info(self.cfg, top1_scores)

        return val_loss, val_accu, val_mse, infos
                        
    def sample(self, epoch, test_db, N, random_or_not=False):
        ##############################################################
        # Output prefix
        ##############################################################
        output_dir = osp.join(self.cfg.model_dir, '%03d'%epoch, 'vis')
        pred_dir   = osp.join(self.cfg.model_dir, '%03d'%epoch, 'pred')
        gt_dir     = osp.join(self.cfg.model_dir, '%03d'%epoch, 'gt')
        img_dir  = osp.join(self.cfg.model_dir, '%03d'%epoch, 'color')

        maybe_create(output_dir); maybe_create(pred_dir)
        maybe_create(gt_dir); maybe_create(img_dir)

        ##############################################################
        # Main loop
        ##############################################################
        plt.switch_backend('agg')
        if random_or_not:
            indices = np.random.permutation(range(len(test_db)))
        else:
            indices = range(len(test_db))
        indices = indices[:N]

        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        for i in indices:
            entry    = test_db[i]
            gt_scene = test_db.scenedb[i]

            gt_img = cv2.imread(entry['color_path'], cv2.IMREAD_COLOR)
            gt_img, _, _ = create_squared_image(gt_img)
            gt_img = cv2.resize(gt_img, (self.cfg.draw_size[0], self.cfg.draw_size[1]))

            ##############################################################
            # Inputs
            ##############################################################
            # input_inds_np = np.array(entry['word_inds'])
            sentence= np.array(entry['bert_inds'])
            input_lens_np = np.array(entry['bert_lens'])

            # input_inds = torch.from_numpy(input_inds_np).long().unsqueeze(0)
            input_lens = torch.from_numpy(input_lens_np).long().unsqueeze(0)
            if self.cfg.cuda:
                input_inds = input_inds.cuda()
                input_lens = input_lens.cuda()

            ##############################################################
            # Inference
            ##############################################################
            self.net.eval()
            with torch.no_grad():
                inf_outs, env = net.inference(input_inds, input_lens, -1, 2.0, 0, None)
            frames = env.batch_redraw(return_sequence=True)[0]
            # _, _, _, _, what_wei, where_wei = inf_outs
            what_wei, where_wei = inf_outs[-2:]
            
            if self.cfg.what_attn:
                what_attn_words = self.decode_attention(
                    input_inds_np, input_lens_np, what_wei.squeeze(0))
            if self.cfg.where_attn > 0:
                where_attn_words = self.decode_attention(
                    input_inds_np, input_lens_np, where_wei.squeeze(0))
        
            ##############################################################
            # Draw
            ##############################################################
            fig = plt.figure(figsize=(60, 30))
            plt.suptitle(entry['sentence'], fontsize=50)
            for j in range(frames.shape[0]):
                # print(attn_words[j])
                subtitle = ''
                if self.cfg.what_attn:
                    subtitle = subtitle + ' '.join(what_attn_words[j])
                if self.cfg.where_attn > 0:
                    subtitle = subtitle + '\n' + ' '.join(where_attn_words[j])

                plt.subplot(4, 4, j+1)
                plt.title(subtitle, fontsize=30)
                plt.imshow(frames[j, :, :, ::-1])
                plt.axis('off')
            plt.subplot(4, 4, 16)
            plt.imshow(gt_img[:, :, ::-1])
            plt.axis('off')

            name = osp.splitext(osp.basename(entry['color_path']))[0]
            out_path = osp.join(output_dir, name+'.png')
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
            
            cv2.imwrite(osp.join(pred_dir, name+'.png'), frames[-1])
            cv2.imwrite(osp.join(img_dir,  name+'.png'), gt_img)
            gt_layout = self.db.render_scene_as_output(gt_scene, False, gt_img)
            cv2.imwrite(osp.join(gt_dir, name+'.png'), gt_layout)
            print('sampling: %d, %d'%(epoch, i))

    def show_metric(self, epoch, test_db, N, random_or_not=False):
        ##############################################################
        # Output prefix
        ##############################################################
        output_dir = osp.join(self.cfg.model_dir, '%03d'%epoch, 'metric')
        maybe_create(output_dir)
        ##############################################################
        # Main loop
        ##############################################################
        plt.switch_backend('agg')
        if random_or_not:
            indices = np.random.permutation(range(len(test_db)))
        else:
            indices = range(len(test_db))
        indices = indices[:N]
        test_db.cfg.sent_group=1

        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net

        ev = evaluator(self.db)

        for i in indices:
            entry    = test_db[i]
            gt_scene = test_db.scenedb[i]
            scene_idx = int(gt_scene['img_idx'])
            name = osp.splitext(osp.basename(entry['color_path']))[0]

            ##############################################################
            # Inputs
            ##############################################################
            input_inds_np = np.array(entry['word_inds'])
            input_lens_np = np.array(entry['word_lens'])

            input_inds = torch.from_numpy(input_inds_np).long().unsqueeze(0)
            input_lens = torch.from_numpy(input_lens_np).long().unsqueeze(0)
            if self.cfg.cuda:
                input_inds = input_inds.cuda()
                input_lens = input_lens.cuda()

            ##############################################################
            # Inference
            ##############################################################
            self.net.eval()
            with torch.no_grad():
                inf_outs, env = self.net.inference(input_inds, input_lens, -1, 2.0, 0, None)
            frame = env.batch_redraw(return_sequence=False)[0][0]
            raw_pred_scene = env.scenes[0]

            pred_inds = deepcopy(raw_pred_scene['out_inds'])
            pred_inds = np.stack(pred_inds, 0)
            pred_scene = self.db.output_inds_to_scene(pred_inds)

            graph_1 = scene_graph(self.db, pred_scene, None, False)
            graph_2 = scene_graph(self.db, gt_scene,   None, False)

            color_1 = frame.copy()
            gt_img = cv2.imread(entry['color_path'], cv2.IMREAD_COLOR)
            gt_img, _, _ = create_squared_image(gt_img)
            gt_img = cv2.resize(gt_img, (self.cfg.draw_size[0], self.cfg.draw_size[1]))
            color_2 = gt_img

            cv2.imwrite('%09d_b.png'%i, color_1)
            cv2.imwrite('%09d_i.png'%i, color_2)

            color_1 = visualize_unigram(self.cfg, color_1, graph_1.unigrams, (225, 0, 0))
            color_2 = visualize_unigram(self.cfg, color_2, graph_2.unigrams, (225, 0, 0))
            color_1 = visualize_bigram(self.cfg, color_1, graph_1.bigrams, (0, 0, 255))
            color_2 = visualize_bigram(self.cfg, color_2, graph_2.bigrams, (0, 0, 255))

            scores = ev.evaluate_graph(graph_1, graph_2)

            color_1 = visualize_unigram(self.cfg, color_1, ev.common_pred_unigrams, (0, 225, 0))
            color_2 = visualize_unigram(self.cfg, color_2, ev.common_gt_unigrams,   (0, 225, 0))
            color_1 = visualize_bigram(self.cfg, color_1, ev.common_pred_bigrams, (0, 255, 255))
            color_2 = visualize_bigram(self.cfg, color_2, ev.common_gt_bigrams, (0, 255, 255))

            info = eval_info(self.cfg, scores[None, ...])

            plt.switch_backend('agg')
            fig = plt.figure(figsize=(16, 10))
            title = entry['sentence']
            title += 'UR:%f,UP:%f,BR:%f,BP:%f\n'%(info.unigram_R()[0], info.unigram_P()[0], info.bigram_R()[0], info.bigram_P()[0])
            title += 'scale: %f, ratio: %f, coord: %f, b:%f \n'%(info.scale()[0], info.ratio()[0], info.unigram_coord()[0], info.bigram_coord()[0])

            plt.suptitle(title)
            plt.subplot(1, 2, 1); plt.imshow(color_1[:,:,::-1]); plt.axis('off')
            plt.subplot(1, 2, 2); plt.imshow(color_2[:,:,::-1]); plt.axis('off')

            out_path = osp.join(output_dir, name +'.png')
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)

    def sample_all_top1(self, test_db):
        ##############################################################
        # Output prefix
        ##############################################################
        out_dir = osp.join(self.cfg.model_dir, 'top1_scenes')
        maybe_create(out_dir)
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net
        indices = range(len(test_db))
        scores = []
        for G in range(5):
            test_db.cfg.sent_group = G
            G_dir = osp.join(out_dir, '%02d'%G)
            maybe_create(G_dir)
            img_dir = osp.join(G_dir, 'images')
            maybe_create(img_dir)
            scene_dir = osp.join(G_dir, 'scenes')
            maybe_create(scene_dir)

            for i in indices:
                entry    = test_db[i]
                gt_scene = test_db.scenedb[i]
                ##############################################################
                # Inputs
                ##############################################################
                input_inds_np = np.array(entry['word_inds'])
                input_lens_np = np.array(entry['word_lens'])

                input_inds = torch.from_numpy(input_inds_np).long().unsqueeze(0)
                input_lens = torch.from_numpy(input_lens_np).long().unsqueeze(0)
                if self.cfg.cuda:
                    input_inds = input_inds.cuda()
                    input_lens = input_lens.cuda()

                ##############################################################
                # Inference
                ##############################################################
                self.net.eval()
                with torch.no_grad():
                    inf_outs, env = net.inference(input_inds, input_lens, -1, 2.0, 0, None)

                # for j in range(len(env.scenes)):
                #     bar_scene = env.scenes[j]
                #     bar_inds = bar_scene['out_inds']
                #     if bar_inds[0][0] > self.cfg.EOS_idx:
                #         break
                # if bar_inds[0][0] <= self.cfg.EOS_idx:
                #     continue
                # pred_scene = deepcopy(bar_scene)
                pred_scene = env.scenes[0]


                ##############################################################
                # Evaluate
                ##############################################################
                pred_scores = env.evaluate_scene(pred_scene, gt_scene)
                scores.append(pred_scores)

                ##############################################################
                # Draw
                ##############################################################
                frame = env.batch_redraw(return_sequence=False)[0][0]

                ##############################################################
                # Save scene
                ##############################################################
                pred_inds = deepcopy(pred_scene['out_inds'])
                pred_inds = np.stack(pred_inds, 0)
                foo = test_db.output_inds_to_scene(pred_inds)
                out_scene = {}
                out_scene['boxes'] = [x.tolist() for x in foo['boxes'].astype(np.float64)]
                out_scene['clses'] = foo['clses'].astype(np.int64).tolist()
                out_scene['caption'] = entry['sentence']
                out_scene['width']   = int(gt_scene['width'])
                out_scene['height']  = int(gt_scene['height'])
                img_idx = int(gt_scene['img_idx'])
                out_scene['img_idx'] = img_idx
                scene_path = osp.join(scene_dir, '%02d_'%G+str(img_idx).zfill(12)+'.json')
                img_path = osp.join(img_dir, '%02d_'%G+str(img_idx).zfill(12)+'.jpg')
                cv2.imwrite(img_path, frame)
                with open(scene_path, 'w') as fp:
                    json.dump(out_scene, fp, indent=4, sort_keys=True)
                print(G, i, img_idx)

                # if len(scores) > 5:
                #     break

        scores = np.stack(scores, 0).astype(np.float64)
        infos = eval_info(self.cfg, scores)
        info_path = osp.join(out_dir, 'eval_info_top1.json')
        log_coco_scores(infos, info_path)

    def sample_demo(self, input_sentences):
        output_dir = osp.join(self.cfg.model_dir, 'bert_layout_samples')
        print(output_dir)
        maybe_create(output_dir)
        num_sents = len(input_sentences)
        ##############################################################
        # Main loop
        ##############################################################
        plt.switch_backend('agg')
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net
        for i in range(num_sents):
        # for i in range(5):
            sentence = input_sentences[i]
            ##############################################################
            # Inputs
            ##############################################################
            word_inds, word_lens = self.db.encode_sentence(sentence)
            input_inds_np = np.array(word_inds)
            input_lens_np = np.array(word_lens)
            input_inds = torch.from_numpy(input_inds_np).long().unsqueeze(0)
            input_lens = torch.from_numpy(input_lens_np).long().unsqueeze(0)
            if self.cfg.cuda:
                input_inds = input_inds.cuda()
                input_lens = input_lens.cuda()
            ##############################################################
            # Inference
            ##############################################################
            self.net.eval()
            with torch.no_grad():
                inf_outs, env = net.inference(input_inds, input_lens, -1, 2.0, 0, None)
            frames = env.batch_redraw(return_sequence=True)[0]

            _, objs = torch.max(inf_outs[0], -1)
            objs = objs[0].cpu().data
            print('------------{}------------'.format(i))
            for k in range(len(objs)):
                if objs[k] <= self.cfg.EOS_idx:
                    break
                
                print(self.db.classes[objs[k]])
                print(inf_outs[1][0][k])
            # _, _, _, _, what_wei, where_wei = inf_outs
            # if self.cfg.what_attn:
            #     what_attn_words = self.decode_attention(
            #         input_inds_np, input_lens_np, what_wei.squeeze(0))
            # if self.cfg.where_attn > 0:
            #     where_attn_words = self.decode_attention(
            #         input_inds_np, input_lens_np, where_wei.squeeze(0))
            ##############################################################
            # Draw
            ##############################################################
            fig = plt.figure(figsize=(60, 40))
            plt.suptitle(sentence, fontsize=40)
            for j in range(frames.shape[0]):
                # subtitle = ''
                # if self.cfg.what_attn:
                #     subtitle = subtitle + ' '.join(what_attn_words[j])
                # if self.cfg.where_attn > 0:
                #     subtitle = subtitle + '\n' + ' '.join(where_attn_words[j])
                plt.subplot(4, 3, j+1)
                # plt.title(subtitle, fontsize=30)
                plt.imshow(frames[j, :, :, ::-1])
                plt.axis('off')
            out_path = osp.join(output_dir, '%09d.png'%i)
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)

    def decode_attention(self, word_inds, word_lens, att_logits):
        _, att_inds  = torch.topk(att_logits, 3, -1)
        att_inds  = att_inds.cpu().data.numpy()

        if len(word_inds.shape) > 1:
            lin_inds = []
            for i in range(word_inds.shape[0]):
                lin_inds.extend(word_inds[i, : word_lens[i]].tolist())
            vlen = len(lin_inds)
            npad = self.cfg.max_input_length * 3 - vlen
            lin_inds = lin_inds + [0] * npad
            # print(lin_inds)
            lin_inds = np.array(lin_inds).astype(np.int32)
        else:
            lin_inds = word_inds.copy()
        
        slen, _ = att_inds.shape
        attn_words = []
        for i in range(slen):
            w_inds = [lin_inds[x] for x in att_inds[i]]
            w_strs = [self.db.lang_vocab.index2word[x] for x in w_inds]
            attn_words = attn_words + [w_strs]
        
        return attn_words

    def save_checkpoint(self, epoch, log):
        print(" [*] Saving checkpoints...")
        if self.cfg.cuda and self.cfg.parallel:
            net = self.net.module
        else:
            net = self.net
        checkpoint_dir = osp.join(self.cfg.model_dir, 'bert_layout_ckpts')
        if not osp.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        model_name = "ckpt-%03d-%.4f-%.4f.pkl" % (epoch, log[0], log[1])
        bert_name = "bert-ckpt-%03d-%.4f-%.4f.pkl" % (epoch, log[0], log[1])

        print('saving ckpt to ', checkpoint_dir)
        state = {'net':net.state_dict(), 'optimizer':self.optimizer.optimizer.state_dict(), 'epoch':epoch}
        # torch.save(net.state_dict(), osp.join(checkpoint_dir, model_name))
        torch.save(state, osp.join(checkpoint_dir, model_name))

        print('saving bert to ', checkpoint_dir)
        bert_state = {'net':net.text_encoder.embedding.state_dict(), 'optimizer':self.bert_optimizer.state_dict(), 'epoch':epoch}
        torch.save(bert_state, osp.join(checkpoint_dir, bert_name))
        # torch.save(net.text_encoder.embedding.state_dict(), osp.join(checkpoint_dir, bert_name))

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

class BucketSampler(Sampler):
    def __init__(self, data_source,  
                bucket_boundaries, batch_size=8,):

        data_source.cfg.prepare = True
        ind_n_len = []
        for i in tqdm(range(len(data_source))):
            p = data_source[i]
            ind_n_len.append( (i, p['obj_cnt'].item()) )

        # for i, p in tqdm(enumerate(data_source)):
        #     ind_n_len.append( (i, p['obj_cnt'].item()) )
        self.ind_n_len = ind_n_len
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size
        data_source.cfg.prepare = False
        
        
    def __iter__(self):
        data_buckets = dict()
        # where p is the id number and seq_len is the length of this id number. 
        for p, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(p,seq_len)
            if pid in data_buckets.keys():
                data_buckets[pid].append(p)
            else:
                data_buckets[pid] = [p]

        for k in data_buckets.keys():

            data_buckets[k] = np.asarray(data_buckets[k])

        iter_list = []
        for k in data_buckets.keys():
            np.random.shuffle(data_buckets[k])
            iter_list += (np.array_split(data_buckets[k]
                           , int(data_buckets[k].shape[0]/self.batch_size)))
        shuffle(iter_list) # shuffle all the batches so they arent ordered by bucket
        # size
        for i in iter_list: 
            yield i.tolist() # as it was stored in an array
    
    def __len__(self):
        return len(self.data_source)
    
    def element_to_bucket_id(self, x, seq_length):
        boundaries = list(self.bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        conditions_c = np.logical_and(
          np.less_equal(buckets_min, seq_length),
          np.less(seq_length, buckets_max))
        bucket_id = np.min(np.where(conditions_c))
        return bucket_id

