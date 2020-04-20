# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import Parameter
# import numpy as np

class BERT_SPC(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        # self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.dense = nn.Linear(opt.bert_dim*opt.max_seq_len, opt.polarities_dim)
        self.attn=nn.TransformerEncoderLayer(d_model=opt.bert_dim,
                                            nhead=3,
                                            # dim_feedforward=4*opt.bert_dim,
                                            dropout=0.1)
        # map into 3 space
        # self.emb = nn.Embedding(args.tok_size, args.emb_dim, padding_idx=0)
        # self.emb = nn.Embedding(args.tok_size, args.emb_dim, padding_idx=0)
        self.W = Parameter(torch.rand(opt.batch_size,opt.max_len,))
        self.H = Parameter(torch.rand(*H_size))
        

    def forward(self, inputs):
        text_spc_bert_indices, bert_segments_ids = inputs[0], inputs[1]                              
        encoded_layers, pooled_output , attention = self.bert(text_spc_bert_indices, bert_segments_ids)
        new_encoded_layers=self.attn(encoded_layers)

        new_encoded_layers=



        pooled_output=new_encoded_layers
        logits = self.dense(pooled_output) 
        return logits

