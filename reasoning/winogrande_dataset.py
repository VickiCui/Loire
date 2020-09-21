# coding=utf-8

""" Utility for finetuning BERT/RoBERTa models on WinoGrande. """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
import json


from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
from pytorch_transformers import RobertaTokenizer
import numpy as np
import jsonlines, os
from tqdm import tqdm
import torch
import os.path as osp

Tokenizers = {
    'bert': BertTokenizer, 
    'roberta': RobertaTokenizer}


class WinoGrande(Dataset):
    """Processor for the CommonsenseQA data set.
    exapmle:
    {
        "qID": "3FCO4VKOZ4BJQ6IFC0VAIBK4KTWE7U-2", 
        "sentence": "Sarah was a much better surgeon than Maria so _ always got the easier cases.", 
        "option1": "Sarah", 
        "option2": "Maria", 
        "answer": "2"
    }
    """

    SIZE_SPLITS = ['xs', 's', 'm', 'l', 'xl']
    LABELS = ['1', '2']

    FILE = {'train':'train_{split}.jsonl', 
            'val':'dev.jsonl', 
            'test':'test.jsonl'}

    def __init__(self, config, split=None):
        self.max_len = 0
        self.cfg = config
       
        for k,v in Tokenizers.items():
            if self.cfg.model.lower().startswith(k):
                tokenizer = v
                break
        pretrained_weights = osp.join(self.cfg.pretrained_lm_path, self.cfg.model.lower())

        self.tokenizer = tokenizer.from_pretrained(pretrained_weights, do_lower_case=True)

        # self.tokenizer = BertTokenizer.from_pretrained(os.path.join(self.cfg.bert_path, self.VOCAB), do_lower_case=True)

        if self.cfg.train_size not in self.SIZE_SPLITS:
            size_splits = 'xl'
        else:
            size_splits = self.cfg.train_size

        self.split = split
        if self.split == 'test' and self.cfg.test_file is not None:
            print('preparing data from {}...'.format(self.cfg.test_file))
            self.dataset, self.index2qid = self._create_examples(self.read_jsonl(self.cfg.test_file))
        else:
            if self.split == 'train':
                self.file_name = self.FILE[split].format(split=size_splits)
            else:
                self.file_name = self.FILE[split]
            print('preparing data from {}...'.format(self.file_name))
            self.dataset, self.index2qid = self._create_examples(self.read_jsonl(os.path.join(self.cfg.data_dir, self.file_name)))
        print(self.max_len)
        # exit()

    def read_jsonl(self, input_file):
        """Reads a JSON Lines file."""
        with open(input_file, "r") as f:
            data = [item for item in jsonlines.Reader(f)]
            return data

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, lines):
        with torch.no_grad():
            if self.cfg.feature:
                fea_tokenizer = BertTokenizer.from_pretrained(osp.join(self.cfg.pretrained_lm_path, 'bert-base'), do_lower_case=True)
                feature = BertModel.from_pretrained(osp.join(self.cfg.pretrained_lm_path, 'bert-base'))   
                if self.cfg.pretrained_bert is not None:
                    print('loading feature ckpt from ',self.cfg.pretrained_bert)   
                    assert osp.exists(self.cfg.pretrained_bert)
                    if self.cfg.cuda:
                        feature = feature.cuda()
                        checkpoint = torch.load(self.cfg.pretrained_bert) 
                    else:
                        checkpoint = torch.load(self.cfg.pretrained_bert, map_location=lambda storage, loc: storage)
                    feature.load_state_dict(checkpoint['net'])

            examples = []
            index2qid = []
            i = 0
            if self.cfg.test:
                lines = lines[:200]
            for line in tqdm(lines):
                data = dict()
                data['index'] = i
                i += 1

                data['qid'] = line['qID']
                index2qid.append(data['qid'])

                sentence = line['sentence']

                name1 = line['option1']
                name2 = line['option2']

                # data['sentence'] = line['sentence']
                # data['option1'] = line['option1']
                # data['option2'] = line['option2']

                conj = "_"
                idx = sentence.index(conj)
                context = sentence[:idx]
                option_str = "_ " + sentence[idx + len(conj):].strip()

                option1 = option_str.replace("_", name1)
                option2 = option_str.replace("_", name2)

                options=[
                        {
                            'segment1': context,
                            'segment2': option1
                        },
                        {
                            'segment1': context,
                            'segment2': option2
                        }
                    ] 

                # the test set has no answer key so use '1' as a dummy label
                data['label_ids'] = self.LABELS.index(line.get('answer', '1'))

                _, data['token_ids'], data['mask'], data['segment_ids'] = self.example_to_token_ids_segment_ids_label_ids(options,self.tokenizer,
                                                                cls_token_at_end=False,
                                                                cls_token=self.tokenizer.cls_token,
                                                                sep_token=self.tokenizer.sep_token,
                                                                sep_token_extra=False,
                                                                cls_token_segment_id=0,
                                                                pad_on_left=False,
                                                                pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                                                                pad_token_segment_id=0
                                                            )
                
                if self.cfg.feature:
                    if self.cfg.model=='bert':
                        input_ids = torch.Tensor(data['token_ids']).long()
                        input_mask = torch.Tensor(data['mask']).long()
                        segment_ids = torch.Tensor(data['segment_ids']).long()
                    else:
                        _, input_ids, segment_ids, input_mask = self.example_to_token_ids_segment_ids_label_ids(
                                                                options, tokenizer=fea_tokenizer,
                                                                cls_token_at_end=False,
                                                                cls_token=fea_tokenizer.cls_token,
                                                                sep_token=fea_tokenizer.sep_token,
                                                                cls_token_segment_id=0,
                                                                pad_on_left=False,
                                                                pad_token=fea_tokenizer.convert_tokens_to_ids([fea_tokenizer.pad_token])[0],
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

    def example_to_token_ids_segment_ids_label_ids(self, options, tokenizer,cls_token_at_end=False, pad_on_left=False,
                                                 cls_token='[CLS]', sep_token='[SEP]', sep_token_extra=False, pad_token=0,
                                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                                 mask_padding_with_zero=True):

        res_tokens, res_input_ids, res_input_mask, res_segment_ids = [], [], [], []
        for option in options:
            context_tokens = tokenizer.tokenize(option['segment1'])
            option_tokens = tokenizer.tokenize(option['segment2'])

            special_tokens_count = 4 if sep_token_extra else 3
            self._truncate_seq_pair(context_tokens, option_tokens, self.cfg.max_seq_length - special_tokens_count)

            tokens = context_tokens + [sep_token]

            # if sep_token_extra:
            #     # roberta uses an extra separator b/w pairs of sentences
            #     tokens += [sep_token]

            segment_ids = [sequence_a_segment_id] * len(tokens)
            tokens += option_tokens + [sep_token]

            segment_ids += [sequence_b_segment_id] * (len(option_tokens) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # print(tokens)
            # print(input_ids)

            self.max_len = max(self.max_len, len(input_ids))

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = self.cfg.max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == self.cfg.max_seq_length
            assert len(input_mask) == self.cfg.max_seq_length
            assert len(segment_ids) == self.cfg.max_seq_length

            res_tokens.append(tokens)
            res_input_ids.append(input_ids)
            res_input_mask.append(input_mask)
            res_segment_ids.append(segment_ids)
            # print(tokens)
            # print('-------------------')
            # print(input_ids)
            # print('-------------------')
            # print(input_mask)
            # print('-------------------')
            # print(segment_ids)
            # exit()

        return res_tokens, res_input_ids, res_input_mask, res_segment_ids

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        entry = {}
        data = self.dataset[idx].copy()
        entry['token_ids'] = np.array(data['token_ids'])
        entry['segment_ids'] = np.array(data['segment_ids'])
        entry['label_ids'] = np.array(data['label_ids'])
        entry['mask'] = np.array(data['mask'])
        entry['index'] = np.array(data['index'])
        if self.cfg.feature:
            entry['feature'] = np.array(data['feature'])
            entry['fea_mask'] = np.array(data['fea_mask'])

        return entry

