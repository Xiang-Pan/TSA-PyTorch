# -*- coding: utf-8 -*-
# file: BERT_ASPECT.py
# author: xiangpan <xiangpan.cs@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn


class BERT_COMPETE(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_COMPETE, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        # 'bert_compete':['bert_compete_cls_pos','bert_compete_indices','bert_compete_segments_ids']
        bert_compete_cls_pos,bert_compete_indices,bert_compete_segments_ids = inputs[0],inputs[1],inputs[2]
        # print(bert_compete_indices)
        word_output, pooled_output = self.bert(bert_compete_indices, bert_compete_segments_ids)
        # print('pooled_output',pooled_output.shape)
        # print('word_output',word_output.shape)
        # pooled_output = self.dropout(pooled_output)
        # cls_eb=word_output[:,bert_compete_cls_pos,:]
        # print('bert_compete_cls_pos',bert_compete_cls_pos.shape)
        # print(len(word_output))

        # cls_eb=word_output[range(len(word_output)),bert_compete_cls_pos,:]

        # print('cls_eb',cls_eb.shape)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        # print(logits.shape)
        return logits
