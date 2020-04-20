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
        self.opt=opt
        # self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.dense = nn.Linear(opt.bert_dim*opt.max_seq_len, opt.polarities_dim)
        
        # self.multihead_attn = nn.MultiheadAttention(opt.bert_dim, num_heads)
        self.attn=nn.TransformerEncoderLayer(d_model=opt.bert_dim,
                                            nhead=3,
                                            # dim_feedforward=4*opt.bert_dim,
                                            dropout=0.1)
        # map into 3 space
        

    def forward(self, inputs):
        text_spc_bert_indices, bert_segments_ids = inputs[0], inputs[1]                              
        encoded_layers, pooled_output , attention = self.bert(text_spc_bert_indices, bert_segments_ids)
        print(len(attention))
        print(attention[8].shape)
        # new_encoded_layers=self.attn(encoded_layers)
        new_encoded_layers=torch.reshape(encoded_layers, (-1, self.opt.bert_dim*self.opt.max_seq_len))
        # pooled_output = self.dropout(pooled_output)
        # pooled_output = self.dropout(new_encoded_layers[:,0,:])
        # pooled_output=new_encoded_layers[:,0,:]
        pooled_output=new_encoded_layers
        logits = self.dense(pooled_output) 
        return logits

