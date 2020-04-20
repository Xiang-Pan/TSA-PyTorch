# -*- coding: utf-8 -*-
# file: BERT_ASPECT.py
# author: xiangpan <xiangpan.cs@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn


class BERT_MULTITASK(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_MULTITASK, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        # 'bert_compete':['bert_compete_cls_pos','bert_compete_indices','bert_compete_segments_ids']
        bert_compete_cls_pos,bert_compete_indices,bert_compete_segments_ids = inputs[0],inputs[1],inputs[2]
        word_output, pooled_output = self.bert(bert_compete_indices, bert_compete_segments_ids)
        cls_eb=word_output[range(len(word_output)),bert_compete_cls_pos,:]
        pooled_output = self.dropout(cls_eb)
        logits = self.dense(pooled_output)
        return logits
