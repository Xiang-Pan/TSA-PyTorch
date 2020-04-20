# -*- coding: utf-8 -*-
# file: BERT_ASPECT.py
# author: xiangpan <xiangpan.cs@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn


class BERT_TARGET(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_TARGET, self).__init__()
        self.bert = bert
        self.opt=opt
        
        self.max_pool= nn.MaxPool1d(1)
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.attn=nn.TransformerEncoderLayer(d_model=opt.bert_dim,
                                            nhead=3,
                                            # dim_feedforward=4*opt.bert_dim,
                                            dropout=0.1)
                                            
    def forward(self, inputs):
        bert_aspect_indices,bert_aspect_segments_ids,target_begin = inputs[0],inputs[1],inputs[2]
        # bert_aspect_indices,bert_aspect_segments_ids= inputs[0],inputs[1]
        # print(aspect_in_text[0])
        # print(target_begin.shape)
        word_output, pooled_output,attention = self.bert(bert_aspect_indices, bert_aspect_segments_ids)
        # word_output=self.attn(word_output)
        cls_eb=word_output[range(len(word_output)),target_begin,:]
        
        pooled_output = self.dropout(cls_eb)
        logits = self.dense(pooled_output)
        return logits
