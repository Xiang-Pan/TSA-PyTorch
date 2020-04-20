# -*- coding: utf-8 -*-
# file: BERT_KG.py
# author: xiangpan <xiangpan.cs@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
# import sys

# sys.path.append('./knowledge_bert')
# from models.knowledge_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForTokenClassification


#     """BERT model for token-level classification.
#     This module is composed of the BERT model with a linear layer on top of
#     the full hidden state of the last layer.

#     Params:
#         `config`: a BertConfig class instance with the configuration to build a new model.
#         `num_labels`: the number of classes for the classifier. Default = 2.

#     Inputs:
#         `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
#             with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
#             `extract_features.py`, `run_classifier.py` and `run_squad.py`)
#         `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
#             types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
#             a `sentence B` token (see BERT paper for more details).
#         `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
#             selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
#             input sequence length in the current batch. It's the mask that we typically use for attention when
#             a batch has varying length sentences.
#         `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
#             with indices selected in [0, ..., num_labels].

#     Outputs:
#         if `labels` is not `None`:
#             Outputs the CrossEntropy classification loss of the output with the labels.
#         if `labels` is `None`:
#             Outputs the classification logits of shape [batch_size, sequence_length, num_labels].

#     Example usage:
#     ```python
#     # Already been converted into WordPiece token ids
#     input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
#     input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
#     token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

#     config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
#         num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

#     num_labels = 2

#     model = BertForTokenClassification(config, num_labels)
#     logits = model(input_ids, token_type_ids, input_mask)
# """


class BERT_KG(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_KG, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        _, pooled_output = self.bert(text_bert_indices, bert_segments_ids)
        print(pooled_output.size())
        pooled_output = self.dropout(pooled_output)
        print(pooled_output.size())
        logits = self.dense(pooled_output)
        return logits

# class BERT_KG(nn.Module):
#     def __init__(self, bert, opt):
#         super(BERT_KG, self).__init__()
#         self.bert_kg = bert
#         # self.dropout = nn.Dropout(opt.dropout)
#         # self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

#     def forward(self, inputs):
#         text_bert_indices, bert_segments_ids, input_mask = inputs[0], inputs[1], inputs[2]
#         input_ids = text_bert_indices
#         token_type_ids = bert_segments_ids
#         logits = self.bert_kg(input_ids, token_type_ids,input_mask)
#         return logits
        # text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        # _, pooled_output = self.bert(text_bert_indices, bert_segments_ids)
        # print(pooled_output.size())
        # pooled_output = self.dropout(pooled_output)
        # print(pooled_output.size())
        # logits = self.dense(pooled_output)
        # return logits


# class BERT_KG(nn.Module):
#     def __init__(self, bert, opt):
#         super(BERT_KG, self).__init__()
#         # config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
#         # segments_ids = [1] * 10 + [0] * 2
#         # BertForTokenClassification.from_pretrained('ernie_base')
#         self.input_mask=[[1]*opt.max_seq_len]*opt.batch_size
#         model, _ = BertForTokenClassification.from_pretrained('ernie_base')
#         model.num_labels = 3
#         self.bert_kg = model
#         # self.num_labels = 3
#         # self.dropout = nn.Dropout(opt.dropout)
#         # self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

#     def forward(self, inputs):
#         text_bert_indices, bert_segments_ids, input_mask = inputs[0], inputs[1], inputs[2]
#         input_ids = text_bert_indices
#         token_type_ids = bert_segments_ids
#         logits = self.bert_kg(input_ids, token_type_ids,input_mask)
#         return logits
