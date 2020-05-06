# -*- coding: utf-8 -*-
# file: BERT_ASPECT.py
# author: xiangpan <xiangpan.cs@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
from layers.attention import Attention
from layers.attention import TransformerEncoderLayer

class BERT_MULTI_TARGET(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_MULTI_TARGET, self).__init__()
        
        
        
        self.bert = bert
        self.opt=opt
        self.max_pool= nn.MaxPool1d(1)
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

        self.chg_dense = nn.Linear(opt.bert_dim, 2)


        self.tfm=TransformerEncoderLayer(d_model=opt.bert_dim,
                                            nhead=3,
                                            # dim_feedforward=4*opt.bert_dim,
                                            dropout=0.1)
        self.tfm1=TransformerEncoderLayer(d_model=opt.bert_dim,
                                            nhead=3,
                                            # dim_feedforward=4*opt.bert_dim,
                                            dropout=0.1)
        
        self.classifier_concat=nn.Linear(opt.bert_dim*2, 9)
        self.classifier_criterion=nn.CrossEntropyLoss()
        self.chg_criterion=nn.CrossEntropyLoss()

        # self.attn_k = Attention(opt.bert_dim, hidden_dim=2048,out_dim=opt.bert_dim, n_head=1, score_function='dot_product', dropout=opt.dropout)

            # if perturbation is not None:
            #     perturbation.to(self.opt.device)
            #     word_output += perturbation
    def forward(self,inputs,perturbation):
        multi_target_indices,multi_target_segments_ids,target_pos,poss,polarity_list,polarity,isaug = inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5],inputs[6]

        # bert_word_embedder=self.bert.get_input_embeddings()
        # bert_word_eb=bert_word_embedder(multi_target_indices)
        if perturbation is not None:
            perturbation.to(self.opt.device)
        word_output,pooled_output,bert_word_eb,attention, = self.bert(multi_target_indices, multi_target_segments_ids,adv_eb=perturbation)
        # ebs=self.bert.get_input_embeddings()

        # print(ebs.shape)
        bert_word_output=word_output
        reg_can=0

        # can reg
        if 1:
            hc,scores=self.tfm(torch.transpose(word_output,0,1))
            hc=torch.transpose(hc,0,1)
            for i in range(len(scores)):# batch size
                target_pos_i=poss[i].index_select(0,torch.nonzero(poss[i]).reshape(-1))
                M_i=scores[i].index_select(0,target_pos_i)
                # reg_i=torch.norm(torch.matmul(M_i.t(),M_i)-torch.eye(M_i.shape[1]).to(self.opt.device))
                reg_i=torch.norm(torch.matmul(M_i.t(),M_i).fill_diagonal_(0))
                reg_can=reg_i+reg_can
        else:
            hc=word_output

        # aux reg
        reg_aux=0

        out_aux=0
        for i in range(len(polarity_list)):# batch_size  i th batch
            # print(polarity_list)
            tartget_polarity_i=polarity[i]
            target_pos_i=target_pos[i]
            # poss i
            poss_i=poss[i].index_select(0,torch.nonzero(poss[i]).reshape(-1))
            
            # get target eb
            word_output_i=word_output[i]
            target_eb_i=word_output_i[target_pos_i]

            polarity_list_i=polarity_list[i]
            polarity_list_i=polarity_list_i.index_select(0,torch.nonzero(polarity_list_i).reshape(-1))
            polarity_list_i=polarity_list_i-1
            
            regi=0
            for j in range(len(poss_i)):
                # print(j)
                other_pos=poss_i[j]
                other_polarity=polarity_list_i[j]
                regj=0
                # if other_pos== target_pos_i:
                #     other_eb=target_eb_i
                #     aux_cls_logeits=self.classifier_concat(torch.cat((target_eb_i,other_eb),0))
                #     res=torch.tensor([tartget_polarity_i*3+other_polarity]).to(self.opt.device)
                #     out_aux=aux_cls_logeits
                #     aux_cls_logeits = aux_cls_logeits[None, :]
                    
                #     regj=self.classifier_criterion(aux_cls_logeits,res)
                # print('other_pos',other_pos)
                # print('target_pos_i',target_pos_i)
                if other_pos!= target_pos_i:
                    other_eb=word_output_i[other_pos]
                    aux_cls_logeits=self.classifier_concat(torch.cat((target_eb_i,other_eb),0))
                    res=torch.tensor([tartget_polarity_i*3+other_polarity]).to(self.opt.device)
                    aux_cls_logeits = aux_cls_logeits[None, :]
                    # aux_cls_logeits = self.dropout(aux_cls_logeits)
                    regj=self.classifier_criterion(aux_cls_logeits,res)
                    # print(regj)
                regi+=regj
            reg_aux+=regi
        
        cls_eb=hc[range(len(hc)),target_pos,:]
        pooled_output = self.dropout(cls_eb)


        logits = self.dense(pooled_output)

        chg_logits=self.chg_dense(pooled_output)
        reg_chg_loss=self.chg_criterion(chg_logits,isaug)

        # print(reg_aux)
        return out_aux,logits,reg_can,reg_aux,bert_word_eb,reg_chg_loss
