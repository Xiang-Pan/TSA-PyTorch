# -*- coding: utf-8 -*-
# file: BERT_ASPECT.py
# author: xiangpan <xiangpan.cs@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn


# TD-BERT

class BERT_ASPECT(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_ASPECT, self).__init__()
        self.bert = bert
        self.opt=opt
        
        self.max_pool= nn.MaxPool1d(1)
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        bert_aspect_indices,bert_aspect_segments_ids,aspect_in_text,aspect_len = inputs[0],inputs[1],inputs[2],inputs[3]
        # bert_aspect_indices,bert_aspect_segments_ids= inputs[0],inputs[1]
        # print(aspect_in_text[0])

        word_output, pooled_output = self.bert(bert_aspect_indices, bert_aspect_segments_ids)

        all_polled_eb=[]
        for i in range(len(word_output)):
            aspect_eb_i=word_output[i].index_select(0,torch.LongTensor(range(aspect_in_text[i][0].item()+1,aspect_in_text[i][0].item()+1+aspect_len[i])).to(self.opt.device))
            aspect_eb_i=aspect_eb_i.unsqueeze(0)
            max_pooled=self.max_pool(aspect_eb_i)
            max_pooled=max_pooled.squeeze(0)
            all_polled_eb.append(max_pooled[0])
            
        pooled_output = torch.stack(all_polled_eb,0)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits
