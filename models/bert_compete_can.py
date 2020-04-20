# -*- coding: utf-8 -*-
# file: BERT_ASPECT.py
# author: xiangpan <xiangpan.cs@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
from layers.attention import Attention



class BERT_COMPETE(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_COMPETE, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.attn_k = Attention(opt.bert_dim, out_dim=opt.bert_dim, n_head=1, score_function='dot_product', dropout=opt.dropout)

    def forward(self, inputs):
        # 'bert_compete':['bert_compete_cls_pos','bert_compete_indices','bert_compete_segments_ids']
        bert_compete_cls_pos,bert_compete_indices,bert_compete_segments_ids,poss = inputs[0],inputs[1],inputs[2],inputs[3]
        # print(bert_compete_indices)
        encoded_layers, pooled_output , attention = self.bert(bert_compete_indices, bert_compete_segments_ids)
        hc, scores = self.attn_k(encoded_layers, encoded_layers)
        # print(scores.shape)
        # print(poss)
        reg=0
        for i in range(len(scores)):
            M_i=scores[i].index_select(1,poss[i])
            reg_i=torch.norm(torch.matmul(M_i.t(),M_i)-torch.eye(M_i.shape[1]).to('cuda:1'))
            # print(reg_i)
            reg=reg_i+reg
            # print(reg)
            # reg2=M*M.T-
        # M=scores[range(len(scores)),poss,:]
        # print(M.shape)
        # reg2=M*M.T-

        cls_eb=hc[range(len(hc)),bert_compete_cls_pos,:]

        # print('cls_eb',cls_eb.shape)
        pooled_output = self.dropout(cls_eb)
        logits = self.dense(pooled_output)
        # print(logits.shape)
        return logits,reg
