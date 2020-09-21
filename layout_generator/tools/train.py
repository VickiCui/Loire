#!/usr/bin/env python

import _init_paths
import os, sys, cv2, math
import random, json, logz
import numpy as np
import os.path as osp
from copy import deepcopy
import matplotlib.pyplot as plt
from glob import glob

from modules.evaluator import *
from modules.my_trainer import SupervisedTrainer
from datasets.layout_coco import layout_coco_bert, test_layout_coco_bert
from datasets.layout_vg import layout_vg_bert, test_layout_vg_bert
from datasets.vg_coco import layout_vg_coco_bert, test_layout_vg_coco_bert
from layout_utils import *
from layout_config import get_config

dataset = {'coco': [layout_coco_bert, test_layout_coco_bert],
           'vg': [layout_vg_bert, test_layout_vg_bert],
           'vg_coco': [layout_vg_coco_bert, test_layout_vg_coco_bert]}

def train_model(config):    
    transformer = volume_normalize('background')
    if config.test == True:
        layoutdata = dataset[config.data][1]
    else:
        layoutdata = dataset[config.data][0]
    train_db = layoutdata(config, split='train', transform=transformer)   
    val_db   = layoutdata(config, split='val',   transform=transformer) 
    # test_db  = layoutdata(config, split='val',  transform=transformer)

    trainer = SupervisedTrainer(train_db)
    trainer.train(train_db, val_db, val_db)   


if __name__ == '__main__':
    cv2.setNumThreads(0)
    config, unparsed = get_config()
    config.bert = True
    config.if_sample = False
    config = layout_arguments(config)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    train_model(config)
