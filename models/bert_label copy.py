# -*- coding: utf-8 -*-
# file: BERT_RAW.py
# author: xiangpan <xiangpan.cs@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
    # parser.add_argument('--hidden_dim', default=300, type=int)
    # parser.add_argument('--bert_dim', default=768, type=int)
    # parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    # parser.add_argument('--max_seq_len', default=80, type=int)

class BERT_LABEL(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_LABEL, self).__init__()
        self.bert = bert
        # encoder_layer = nn.TransformerEncoderLayer(d_model=opt.bert_dim,
        #                                                  nhead=12,
        #                                                  dim_feedforward=4*opt.bert_dim,
        #                                                  dropout=0.1)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.context_layer=nn.TransformerEncoderLayer(d_model=opt.bert_dim,
                                                         nhead=12,
                                                         dim_feedforward=4*opt.bert_dim,
                                                         dropout=0.1)
        self.label_layer=nn.Embedding(num_embeddings=opt.max_seq_len,
                                        embedding_dim=lebel_layer_dim
                                    )

        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)


    def forward(self, inputs):
        text_raw_bert_indices, bert_segments_ids = inputs[0], inputs[1]                                                     
        encoding, pooled_output = self.bert(text_raw_bert_indices, bert_segments_ids)
        contexted=self.context_layer(encoding)
        contexted
        # contexted_cls=contexted[:,0,:]
        # print(contexted_cls.shape)
        # pr

        logits = self.dense(contexted_cls) 
        return logits
