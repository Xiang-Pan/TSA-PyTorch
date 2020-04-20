# -*- coding: utf-8 -*-
# file: BERT_ASPECT.py
# author: xiangpan <xiangpan.cs@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
import torch.nn.init as init
import math

from pytorch_pretrained_bert.modeling import BertModel,BertEmbeddings,BertPooler,BertEncoder

class VocabGraphConvolution(nn.Module):
    """Vocabulary GCN module.

    Params:
        `voc_dim`: The size of vocabulary graph
        `num_adj`: The number of the adjacency matrix of Vocabulary graph
        `hid_dim`: The hidden dimension after XAW
        `out_dim`: The output dimension after Relu(XAW)W
        `dropout_rate`: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.

    Inputs:
        `vocab_adj_list`: The list of the adjacency matrix
        `X_dv`: the feature of mini batch document, can be TF-IDF (batch, vocab), or word embedding (batch, word_embedding_dim, vocab)

    Outputs:
        The graph embedding representation, dimension (batch, `out_dim`) or (batch, word_embedding_dim, `out_dim`)

    """
    def __init__(self,voc_dim, num_adj, hid_dim, out_dim, dropout_rate=0.2):
        super(VocabGraphConvolution, self).__init__()
        self.voc_dim=voc_dim
        self.num_adj=num_adj
        self.hid_dim=hid_dim
        self.out_dim=out_dim

        for i in range(self.num_adj):
            setattr(self, 'W%d_vh'%i, nn.Parameter(torch.randn(voc_dim, hid_dim)))

        self.fc_hc=nn.Linear(hid_dim,out_dim) 
        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.reset_parameters()

    def reset_parameters(self):
        for n,p in self.named_parameters():
            if n.startswith('W') or n.startswith('a') or n in ('W','a','dense'):
                init.kaiming_uniform_(p, a=math.sqrt(5))

    def forward(self, vocab_adj_list, X_dv, add_linear_mapping_term=False):
        for i in range(self.num_adj):
            H_vh=vocab_adj_list[i].mm(getattr(self, 'W%d_vh'%i))
            # H_vh=self.dropout(F.elu(H_vh))
            H_vh=self.dropout(H_vh)
            H_dh=X_dv.matmul(H_vh)

            if add_linear_mapping_term:
                H_linear=X_dv.matmul(getattr(self, 'W%d_vh'%i))
                H_linear=self.dropout(H_linear)
                H_dh+=H_linear

            if i == 0:
                fused_H = H_dh
            else:
                fused_H += H_dh

        out=self.fc_hc(fused_H)
        return out

class Pretrain_VGCN(nn.Module):
    """Pretrain_VGCN can pre-train the weights of VGCN moduel in the VGCN-BERT. It is also a pure VGCN classification model for word embedding input.

    Params:
        `word_emb`: The instance of word embedding module, we use BERT word embedding module in general.
        `word_emb_dim`: The dimension size of word embedding.
        `gcn_adj_dim`: The size of vocabulary graph
        `gcn_adj_num`: The number of the adjacency matrix of Vocabulary graph
        `hid_dim`: The hidden dimension after XAW
        `gcn_embedding_dim`: The output dimension after Relu(XAW)W
        `num_labels`: the number of classes for the classifier. Default = 2.
        `dropout_rate`: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.

    Inputs:
        `vocab_adj_list`: The list of the adjacency matrix
        `gcn_swop_eye`: The transform matrix for transform the token sequence (sentence) to the Vocabulary order (BoW order)
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).

    Outputs:
        The graph embedding representation, dimension (batch, `out_dim`) or (batch, word_embedding_dim, `out_dim`)

    """
    def __init__(self, word_emb, word_emb_dim, gcn_adj_dim, gcn_adj_num, gcn_embedding_dim, num_labels,dropout_rate=0.2):
        super(Pretrain_VGCN, self).__init__()
        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.word_emb=word_emb
        self.vocab_gcn=VocabGraphConvolution(gcn_adj_dim, gcn_adj_num, 128, gcn_embedding_dim) #192/256
        self.classifier = nn.Linear(gcn_embedding_dim*word_emb_dim, num_labels)
    def forward(self, vocab_adj_list, gcn_swop_eye, input_ids, token_type_ids=None, attention_mask=None):
        words_embeddings = self.word_emb(input_ids)
        vocab_input=gcn_swop_eye.matmul(words_embeddings).transpose(1,2)
        gcn_vocab_out = self.vocab_gcn(vocab_adj_list, vocab_input).transpose(1,2)
        gcn_vocab_out=self.dropout(self.act_func(gcn_vocab_out))
        # import pudb;pu.db
        out=self.classifier(gcn_vocab_out.flatten(start_dim=1))
        return out


class VGCNBertEmbeddings(BertEmbeddings):
    """Construct the embeddings from word, VGCN graph, position and token_type embeddings.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model
        `gcn_adj_dim`: The size of vocabulary graph
        `gcn_adj_num`: The number of the adjacency matrix of Vocabulary graph
        `gcn_embedding_dim`: The output dimension after VGCN

    Inputs:
        `vocab_adj_list`: The list of the adjacency matrix
        `gcn_swop_eye`: The transform matrix for transform the token sequence (sentence) to the Vocabulary order (BoW order)
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.

    Outputs:
        the word embeddings fused by VGCN embedding, position embedding and token_type embeddings.

    """
    def __init__(self, config, gcn_adj_dim, gcn_adj_num, gcn_embedding_dim):
        super(VGCNBertEmbeddings, self).__init__(config)
        assert gcn_embedding_dim>=0
        self.gcn_embedding_dim=gcn_embedding_dim
        self.vocab_gcn=VocabGraphConvolution(gcn_adj_dim, gcn_adj_num, 128, gcn_embedding_dim) #192/256

    def forward(self, vocab_adj_list, gcn_swop_eye, input_ids, token_type_ids=None, attention_mask=None):
        words_embeddings = self.word_embeddings(input_ids)
        vocab_input=gcn_swop_eye.matmul(words_embeddings).transpose(1,2)
        
        if self.gcn_embedding_dim>0:
            gcn_vocab_out = self.vocab_gcn(vocab_adj_list, vocab_input)
         
            gcn_words_embeddings=words_embeddings.clone()
            for i in range(self.gcn_embedding_dim):
                tmp_pos=(attention_mask.sum(-1)-2-self.gcn_embedding_dim+1+i)+torch.arange(0,input_ids.shape[0]).to(input_ids.device)*input_ids.shape[1]
                gcn_words_embeddings.flatten(start_dim=0, end_dim=1)[tmp_pos,:]=gcn_vocab_out[:,:,i]

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        if self.gcn_embedding_dim>0:
            embeddings = gcn_words_embeddings + position_embeddings + token_type_embeddings
        else:
            embeddings = words_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BERT_GCN(BertModel):
    def __init__(self, bert, opt):
        super(BERT_GCN, self).__init__()
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
