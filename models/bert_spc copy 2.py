# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
# import numpy as np

class BERT_SPC(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_spc_bert_indices, bert_segments_ids = inputs[0], inputs[1]                              
        encoded_layers, pooled_output,attention = self.bert(text_spc_bert_indices, bert_segments_ids)
        logits = self.dense(pooled_output) 
        return logits

