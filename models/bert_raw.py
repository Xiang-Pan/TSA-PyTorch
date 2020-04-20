# -*- coding: utf-8 -*-
# file: BERT_RAW.py
# author: xiangpan <xiangpan.cs@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn


class BERT_RAW(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_RAW, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        # self.tfm = nn.TransformerEncoderLayer(  d_model=opt.bert_dim,
        #                                         nhead=12,
        #                                         dim_feedforward=4*opt.bert_dim,
        #                                         dropout=0.1)

    def forward(self, inputs):
        text_raw_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        
                                                        
        encoded_layers, pooled_output = self.bert(text_raw_bert_indices, bert_segments_ids)
        # print(pooled_output)
        # print(pooled_output.size()) # (2,768)
        # aspect_output=  encoded_layers[aspect_indices]
        # aspect_output = self.dropout(aspect_output)
        logits = self.dense(pooled_output) 
        # pooled_output = self.dropout(pooled_output)
        # logits = self.dense(encoded_layers[])
        return logits
