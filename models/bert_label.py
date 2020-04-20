# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
from layers.attention import Attention
from text_models.leam import LEAM

class BERT_LABEL(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_LABEL, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.opt=opt
        # self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.dense = nn.Linear(opt.bert_dim*opt.max_seq_len, opt.polarities_dim)
        self.attn=nn.TransformerEncoderLayer(d_model=opt.bert_dim,
                                            nhead=3,
                                            # dim_feedforward=4*opt.bert_dim,
                                            dropout=0.1)

        self.leam=LEAM(opt.bert_dim)


    def forward(self, inputs):
        # if self.requires_grad_():
        text_spc_bert_indices, bert_segments_ids, targets = inputs[0], inputs[1],  inputs[2] 
        if self.training==False:
            # print(self.requires_grad_())
            targets=None
                              
        encoded_layers, pooled_output , attention = self.bert(text_spc_bert_indices, bert_segments_ids)
        new_encoded_layers=self.attn(encoded_layers)
        logits,reg_loss=self.leam(new_encoded_layers,label=targets)
        # print(att_out.shape)
        # att_out=torch.reshape(att_out, (-1, self.opt.bert_dim*self.opt.max_seq_len))
        # pooled_output = self.dropout(pooled_output)
        # pooled_output = self.dropout(new_encoded_layers[:,0,:])
        # pooled_output=new_encoded_layers[:,0,:]
        # pooled_output=att_out
        # logits = self.dense(pooled_output) 
        return logits,reg_loss

