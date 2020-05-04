# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
from layers.attention import Attention
# from torch_multi_head_attention import MultiHeadAttention


# import numpy as np
# from text_models.leam import LEAM
# import torchsparseattn 

class TD_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.opt=opt
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.attn=nn.TransformerEncoderLayer(d_model=opt.bert_dim,
                                            nhead=3,
                                            # dim_feedforward=4*opt.bert_dim,
                                            dropout=0.1)
        # self.attn_k = Attention(opt.bert_dim, out_dim=opt.bert_dim, n_head=3, score_function='dot_product', dropout=opt.dropout)
        # self.attn_q = Attention(opt.bert_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
                                            


    def forward(self, inputs):
        text_spc_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        # print(self.opt.bert_dim)                            
        encoded_layers, pooled_output ,attention, = self.bert(text_spc_bert_indices, bert_segments_ids)
        # word_output,pooled_output,bert_word_eb,attention
        # encoded_layers
        # new_encoded_layers=self.attn(encoded_layers)

        # hc, scores = self.attn_k(encoded_layers, encoded_layers)
        # for i in range(len(scores)):
        #     M_i=scores[i].index_select(1,poss[i])
        #     reg_i=torch.norm(torch.matmul(M_i.t(),M_i)-torch.eye(M_i.shape[1]).to('cuda:1'))
        #     reg=reg_i+reg
        # # print('hc.shape',hc.shape)
        # print('scores.shape',scores.shape)

        
        # print(att_out.shape)
        # att_out=torch.reshape(att_out, (-1, self.opt.bert_dim*self.opt.max_seq_len))
        pooled_output = self.dropout(pooled_output)
        # pooled_output = self.dropout(new_encoded_layers[:,0,:])
        # pooled_output=new_encoded_layers[:,0,:]
        # pooled_output=hc[:,0,:]

        # pooled_output=att_out
        logits = self.dense(pooled_output) 
        # print(logits.shape)
        return logits

