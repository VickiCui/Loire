from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel, RobertaTokenizer
# from pytorch_transformers import RobertaTokenizer
import numpy as np
import jsonlines, os
from tqdm import tqdm
import torch
import os.path as osp

Tokenizers = {
    'bert': BertTokenizer, 
    'roberta': RobertaTokenizer}


class CommonsenseQA(Dataset):
    """Processor for the CommonsenseQA data set.
    exapmle:
    {
      "answerKey": "A", "id": "1afa02df02c908a558b4036e80242fac", 
      "question": 
      {
        "question_concept": "revolving door", 
        "choices": 
        [
          {"label": "A", "text": "bank"}, 
          {"label": "B", "text": "library"}, 
          {"label": "C", "text": "department store"}, 
          {"label": "D", "text": "mall"}, 
          {"label": "E", "text": "new york"}
        ], 
        "stem": "A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?"
      }
    }
    """

    RAND_SPLITS = ['qtoken', 'rand']
    LABELS = ['A', 'B', 'C', 'D', 'E']

    FILE = {'train':'train_{split}_split.jsonl', 
            'val':'dev_{split}_split.jsonl', 
            'test':'test_{split}_split_no_answers.jsonl'}

    def __init__(self, config, split=None):
        self.cfg = config
        self.max_len = 0

        for k,v in Tokenizers.items():
            if self.cfg.model.lower().startswith(k):
                tokenizer = v
                break
        pretrained_weights = osp.join(self.cfg.pretrained_lm_path, self.cfg.model.lower())

        self.tokenizer = tokenizer.from_pretrained(pretrained_weights, do_lower_case=True)
        self.fea_tokenizer = BertTokenizer.from_pretrained(osp.join(self.cfg.pretrained_lm_path, 'bert-base'), do_lower_case=True)

        if self.cfg.rand_split not in self.RAND_SPLITS:
            rand_splits = 'rand'
        else:
            rand_splits = self.cfg.rand_split

        self.split = split
        if self.split == 'test' and self.cfg.test_file is not None:
            self.dataset, self.index2qid = self._create_examples(self.read_jsonl(self.cfg.test_file))
        else:
            self.file_name = self.FILE[split].format(split=rand_splits)
            self.dataset, self.index2qid = self._create_examples(self.read_jsonl(os.path.join(self.cfg.data_dir, self.file_name)))
        print(self.max_len)
        # exit()


    def read_jsonl(self, input_file):
        """Reads a JSON Lines file."""
        with open(input_file, "r") as f:
            data = [item for item in jsonlines.Reader(f)]
            return data

    def get_labels(self):
        return [0, 1, 2, 3, 4]

    def _create_examples(self, lines):
        with torch.no_grad():
            if self.cfg.feature:
                feature = BertModel.from_pretrained(osp.join(self.cfg.pretrained_lm_path, 'bert-base'))
                if self.cfg.cuda:
                    feature = feature.cuda()      
                if self.cfg.pretrained_bert is not None:
                    print('loading feature ckpt from ',self.cfg.pretrained_bert)   
                    assert osp.exists(self.cfg.pretrained_bert)
                    if self.cfg.cuda:
                        checkpoint = torch.load(self.cfg.pretrained_bert) 
                    else:
                        checkpoint = torch.load(self.cfg.pretrained_bert, map_location=lambda storage, loc: storage)
                    try:
                        feature.load_state_dict(checkpoint['net'])
                    except:
                        feature.load_state_dict(checkpoint)

            examples = []
            index2qid = []
            i = 0
            if self.cfg.test:
                lines = lines[:100]
            for line in tqdm(lines):
                data = dict()
                data['qid'] = line['id']
                index2qid.append(data['qid'])
                data['index'] = i
                i += 1
                # data['question'] = self.tokenizer.convert_to_unicode(line['question']['stem'])
                data['question'] = line['question']['stem']
                data['answer'] = [choice['text'] for choice in sorted(
                                        line['question']['choices'],
                                        key=lambda c: c['label'])
                                ]
                # the test set has no answer key so use 'A' as a dummy label
                data['label_ids'] = self.LABELS.index(line.get('answerKey', 'A'))
                # data['token_ids'], data['segment_ids'], data['mask'] = self.example_to_token_ids_segment_ids_label_ids(data, self.cfg.max_seq_length)
                # data['feature_ids'], data['feature_segment'], data['feature_mask'] = self.example_to_token_ids_segment_ids_label_ids(data, 65)
                
                _, data['token_ids'], data['segment_ids'], data['mask'] = self.example_to_token_ids_segment_ids_label_ids(
                                                                data, self.cfg.max_seq_length, self.tokenizer,
                                                                cls_token_at_end=bool(self.cfg.model.lower() in ['xlnet']),            # xlnet has a cls token at the end
                                                                cls_token=self.tokenizer.cls_token,
                                                                sep_token=self.tokenizer.sep_token,
                                                                sep_token_extra=bool(self.cfg.model.lower() in ['roberta']),
                                                                cls_token_segment_id=2 if self.cfg.model.lower() in ['xlnet'] else 0,
                                                                pad_on_left=bool(self.cfg.model.lower() in ['xlnet']),                 # pad on the left for xlnet
                                                                pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                                                                pad_token_segment_id=4 if self.cfg.model.lower() in ['xlnet'] else 0
                                                            )

                # print(data['token_ids'])
                # print(data['mask'])
                # print(data['segment_ids'])
                # exit()

                
                if self.cfg.feature:
                    if self.cfg.model=='bert':
                        input_ids = torch.Tensor(data['token_ids']).long()
                        input_mask = torch.Tensor(data['mask']).long()
                        segment_ids = torch.Tensor(data['segment_ids']).long()
                    else:
                        _, input_ids, segment_ids, input_mask = self.example_to_token_ids_segment_ids_label_ids(
                                                                data, 65, tokenizer=self.fea_tokenizer,
                                                                cls_token_at_end=False,            # xlnet has a cls token at the end
                                                                cls_token=self.fea_tokenizer.cls_token,
                                                                sep_token=self.fea_tokenizer.sep_token,
                                                                sep_token_extra=False,
                                                                cls_token_segment_id=0,
                                                                pad_on_left=False,                 # pad on the left for xlnet
                                                                pad_token=self.fea_tokenizer.convert_tokens_to_ids([self.fea_tokenizer.pad_token])[0],
                                                                pad_token_segment_id=0
                                                            )
                        input_ids = torch.Tensor(input_ids).long()
                        input_mask = torch.Tensor(input_mask).long()
                        segment_ids = torch.Tensor(segment_ids).long()

                    if self.cfg.cuda:
                        input_ids = input_ids.cuda()
                        input_mask = input_mask.cuda()
                        segment_ids = segment_ids.cuda()

                    bert_outputs = feature(input_ids,
                                    attention_mask=input_mask,
                                    token_type_ids=segment_ids)
                                    
                                    
                    data['feature'] = bert_outputs[0].cpu().data
                    data['fea_mask'] = input_mask.cpu().data
                    
                examples.append(data)
            
            torch.cuda.empty_cache()

        return examples, index2qid
        
    def example_to_token_ids_segment_ids_label_ids(self, data, max_seq_length, tokenizer,
                                                   cls_token_at_end=False, pad_on_left=False,
                                                   cls_token='[CLS]', sep_token='[SEP]', sep_token_extra=False, pad_token=0,
                                                   sequence_a_segment_id=0, sequence_b_segment_id=1,
                                                   cls_token_segment_id=0, pad_token_segment_id=0,
                                                   mask_padding_with_zero=True):
        sep_token_extra=False
        """Converts an ``InputExample`` to token ids and segment ids."""
        question = data['question']
        answers = data['answer']
        if self.cfg.prefix:
            question = 'Q: ' + question
            for i in range(len(answers)):
                answers[i] = 'A: ' + answers[i]
        question_tokens = tokenizer.tokenize(question)
        answers_tokens = map(tokenizer.tokenize, answers)

        token_ids = []
        segment_ids = []
        mask = []
        res_tokens = []
        special_tokens_count = 4 if sep_token_extra else 3
        for _, answer_tokens in enumerate(answers_tokens):
            truncated_question_tokens = question_tokens[
            :max((max_seq_length - special_tokens_count)//2, max_seq_length - (len(answer_tokens) + special_tokens_count))]
            truncated_answer_tokens = answer_tokens[
            :max((max_seq_length - special_tokens_count)//2, max_seq_length - (len(question_tokens) + special_tokens_count))]

            choice_tokens = []
            choice_segment_ids = []
            choice_mask = []
            # choice_tokens.append(cls_token)

            # choice_segment_ids.append(0)
            # choice_mask.append(1)
            for question_token in truncated_question_tokens:
                choice_tokens.append(question_token)
                choice_segment_ids.append(sequence_a_segment_id)
                # choice_mask.append(1)
            choice_tokens.append(sep_token)
            choice_segment_ids.append(sequence_a_segment_id)

            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                choice_tokens.append(sep_token)
                choice_segment_ids.append(sequence_a_segment_id)

            # choice_mask.append(1)
            for answer_token in truncated_answer_tokens:
                choice_tokens.append(answer_token)
                choice_segment_ids.append(sequence_b_segment_id)
                # choice_mask.append(1)
            choice_tokens.append(sep_token)
            choice_segment_ids.append(sequence_b_segment_id)
            # choice_mask.append(1)

            if cls_token_at_end:
                choice_tokens = choice_tokens + [cls_token]
                choice_segment_ids = choice_segment_ids + [cls_token_segment_id]
            else:
                choice_tokens = [cls_token] + choice_tokens
                choice_segment_ids = [cls_token_segment_id] + choice_segment_ids

            choice_token_ids = tokenizer.convert_tokens_to_ids(choice_tokens)
            choice_mask = [1 if mask_padding_with_zero else 0] * len(choice_token_ids)

            self.max_len = max(self.max_len, len(choice_tokens))
            
            padding_length = max_seq_length - len(choice_token_ids)
            if pad_on_left:
                choice_token_ids = ([pad_token] * padding_length) + choice_token_ids
                choice_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + choice_mask
                choice_segment_ids = ([pad_token_segment_id] * padding_length) + choice_segment_ids
            else:
                choice_token_ids = choice_token_ids + ([pad_token] * padding_length)
                choice_mask = choice_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                choice_segment_ids = choice_segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(choice_token_ids) == max_seq_length
            assert len(choice_mask) == max_seq_length
            assert len(choice_segment_ids) == max_seq_length

            # print(choice_tokens)
            # print('-------------------')
            # print(choice_token_ids)
            # print('-------------------')
            # print(choice_mask)
            # print('-------------------')
            # print(choice_segment_ids)
            # exit()

            token_ids.append(choice_token_ids)
            segment_ids.append(choice_segment_ids)
            mask.append(choice_mask)
            res_tokens.append(choice_tokens)

        return res_tokens, token_ids, segment_ids, mask

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        entry = {}
        data = self.dataset[idx].copy()
        entry['token_ids'] = np.array(data['token_ids'])
        entry['segment_ids'] = np.array(data['segment_ids'])
        entry['mask'] = np.array(data['mask'])

        # entry['feature_ids'] = np.array(data['feature_ids'])
        # entry['feature_segment'] = np.array(data['feature_segment'])
        # entry['feature_mask'] = np.array(data['feature_mask'])

        entry['label_ids'] = np.array(data['label_ids'])
        entry['index'] = np.array(data['index'])
        if self.cfg.feature:
            entry['feature'] = np.array(data['feature'])
            entry['fea_mask'] = np.array(data['fea_mask'])

        return entry
