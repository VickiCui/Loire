#!/usr/bin/env python
# codes modified from
#   https://github.com/uvavision/Text2Scene/blob/master/lib/modules/layout_encoder.py

import math, cv2, sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from layout_utils import conv3x3
from transformers import BertModel, BertTokenizer
import os
from torchsummary import summary

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class VolumeEncoder(nn.Module):
    def __init__(self, config):
        self.inplanes = 128
        super(VolumeEncoder, self).__init__()
        self.conv1 = nn.Conv2d(config.output_cls_size, 128, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(128)
        self.relu  = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, 128, 1)
        # self.layer2 = self._make_layer(BasicBlock, 128, 1, stride=2)
        self.layer2 = self._make_layer(BasicBlock, config.n_conv_hidden, 1, stride=2)
        self.upsample = nn.Upsample(size=(config.grid_size[1], config.grid_size[0]), mode='bilinear', align_corners=True)
        # self.linear = nn.Linear(256*7*7, 768)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, stack_vols):
        bsize, slen, fsize, height, width = stack_vols.size()
        inputs = stack_vols.view(bsize * slen, fsize, height, width)

        x = self.conv1(inputs)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # print('x', x.shape)
        x = self.upsample(x)
        
        nsize, fsize, gh, gw = x.size()
        assert(nsize == bsize * slen)
        x = x.view(bsize, slen, fsize, gh, gw)

        return x

class BertEncoder(nn.Module):
    def __init__(self, db):
        super(BertEncoder, self).__init__()

        #self.bert = BertModelfrom_pretrained('bert-base-uncased')
        self.db = db
        self.cfg = db.cfg

        self.bert_path = 'bert-base' if self.cfg.bert_path=='' else self.cfg.bert_path
        self.tokenizer_path = 'bert-base' if self.cfg.bert_path=='' else os.path.join(self.cfg.bert_path, 'bert-base-uncased-vocab.txt')

        self.embedding = BertModel.from_pretrained(self.bert_path, output_hidden_states=True)
        self.dropout = nn.Dropout(self.cfg.emb_dropout_p)
        self.hidden_size = self.cfg.n_src_hidden

        if self.cfg.reduction:
            self.linear = nn.Linear(768, self.hidden_size)
            self.relu  = nn.ReLU(inplace=True)
        

        # if self.cfg.cuda:
        #     self.embedding.cuda()

        
    def forward(self, input_inds, input_lens):
        """
        Args:
            - **input_inds**  (bsize, slen) or (bsize, 3, slen)
            - **input_msks**  (bsize, slen) or (bsize, 3, slen)
        Returns: dict containing
            - **output_feats**   (bsize, tlen, hsize)
            - **output_embed**   (bsize, tlen, esize)
            - **output_msks**    (bsize, tlen)
            - **output_hiddens** [list of](num_layers * num_directions, bsize, hsize)
        """

        bsize, slen = input_inds.size()
        out_cels = []

        tokens, segments, input_masks = [], [], []
        max_len = self.cfg.max_input_length

        tokens = input_inds

        for i in range(bsize):
            curr_len  = input_lens[i][0].data.item()
            # curr_inds = input_inds[i].view(-1)

            # tokenized_text = self.tokenizer.tokenize(curr_inds)
            # indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            # tokens.append(curr_inds)
            segments.append([0] * curr_len)
            input_masks.append([1] * curr_len)

        for j in range(len(tokens)):
            padding = [0] * (max_len - input_lens[j][0].data.item())
            # tokens[j] += padding
            segments[j] += padding
            input_masks[j] += padding

        tokens_tensor = tokens
        segments_tensors = torch.tensor(segments)
        input_masks_tensors = torch.tensor(input_masks)

        if self.cfg.cuda:
            tokens_tensor = tokens_tensor.cuda()
            segments_tensors = segments_tensors.cuda()
            input_masks_tensors = input_masks_tensors.cuda()
            
        bert_outputs = self.embedding(tokens_tensor,
                        attention_mask=input_masks_tensors,
                        token_type_ids=segments_tensors)

        if self.cfg.emb_dropout_p > 0:
            bert_outputs = self.dropout(bert_outputs)
        
        # bert_outputs = self.linear(bert_outputs)
        # bert_outputs = self.relu(bert_outputs)
        out_rfts = bert_outputs[0]
        out_hids = bert_outputs[1]
        out_embs = bert_outputs[2][0]
        
        if self.cfg.reduction:
            out_rfts = self.linear(out_rfts)
            out_hids = self.linear(out_hids)
            out_embs = self.emb_linear(out_embs)

            activation = True
            if activation:
                out_rfts = self.relu(out_rfts)
                out_hids = self.relu(out_hids)
                out_embs = self.relu(out_embs)

        extand_mask = torch.unsqueeze(input_masks_tensors, 2) 
        rfts_mask = extand_mask.expand(-1,-1,self.hidden_size).float()
        embs_mask = extand_mask.expand(-1,-1,self.cfg.n_embed).float()
        out_rfts = out_rfts.mul(rfts_mask)
        out_embs = out_embs.mul(embs_mask)

        inst_msks = np.array(input_masks)
        out_msks = torch.from_numpy(inst_msks).float() 
        if self.cfg.cuda:
            out_rfts = out_rfts.cuda() 
            out_embs = out_embs.cuda() 
            out_msks = out_msks.cuda() 
            out_hids = out_hids.cuda() 

        # print('out_rfts: ', out_rfts.size())
        # print('out_embs: ', out_embs.size())
        # print('out_hids: ', out_hids.size())
        # print('out_msks: ', out_msks.size())

        out = {}
        out['rfts'] = out_rfts
        out['embs'] = out_embs
        out['msks'] = out_msks
        out['hids'] = out_hids
    
        return out['rfts'], out['embs'], out['msks'], out['hids']
