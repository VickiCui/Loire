#!/usr/bin/env python

import argparse
import os.path as osp


this_dir = osp.dirname(__file__)


def str2bool(v):
    return v.lower() in ('true', '1')


parser = argparse.ArgumentParser()

##################################################################
# train paras
##################################################################
parser.add_argument('--cuda', '--gpu', action='store_true')
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--fp16_opt_level',  type=str, default='01')
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--warmup', type=float, default=0.1)
parser.add_argument('--max_grad_norm', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--epochs',  type=int, default=5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for data loader')
parser.add_argument('--log_per_steps', type=int, default=100)
parser.add_argument('--log_dir',  type=str, default=osp.join(this_dir, 'logs'))
parser.add_argument('--tensorboard_logdir',  type=str, default=None)
parser.add_argument('--model_name',  type=str, default='bert')
parser.add_argument('--max_seq_length', type=int, default=128)
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--do_eval', action='store_true')
parser.add_argument('--do_pred', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--accumulation_steps', type=int, default=1)
parser.add_argument('--fix_emb', action='store_true')
parser.add_argument('--ft_last_layer', action='store_true')
parser.add_argument('--weight_decay',  type=float, default=0.01)
parser.add_argument('--adam_eps',  type=float, default=1e-8)
parser.add_argument('--adam_betas', default='(0.9, 0.999)', help='betas for Adam optimizer')
parser.add_argument('--save_ckpt', action='store_true')

##################################################################
# model paras
##################################################################
parser.add_argument('--model',  type=str, default='bert-base') # bert-base / roberta-base / roberta-large
parser.add_argument('--dropout',  type=float, default=0.1, help='dropout rate of bert')
parser.add_argument('--pooler_dropout',  type=float, default=0.0, help='dropout rate of bert')
parser.add_argument('--pretrained', type=str, default=None)
parser.add_argument('--pretrained_bert', type=str, default=None)
parser.add_argument('--pretrained_lm_path', type=str, default='PATH_OF_PRETRAINED_LM')
parser.add_argument('--bert_path', type=str, default=None) # giving the path of bert used for extract visual feature
parser.add_argument('--norm', action='store_true')
parser.add_argument('--feature', action='store_true')
parser.add_argument('--proj', type=str, default='f2t')


##################################################################
# data paras
##################################################################
parser.add_argument('--rand_split',  type=str, default='rand')
parser.add_argument('--data_dir', type=str, default='PATH_OF_DATA') 
parser.add_argument('--test_file', type=str, default=None)
parser.add_argument('--prediction_file', type=str, default='prediction.csv')
parser.add_argument('--task', type=str, default=None) # commonsense_qa / winograde
parser.add_argument('--train_size', type=str, default='xl') # needed for winograde task
parser.add_argument('--prefix', action='store_true', help='prepend a prefix of Q: to the question and A: to the answer')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
