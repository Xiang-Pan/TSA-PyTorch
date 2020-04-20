#coding:utf-8

import sys

import torch
from torch import nn
import torch.nn.functional as F

from models.model import Model
from modules.embedding.embedding import TokenEmbedding
from modules.encoder.lstm_encoder import LstmEncoderLayer
from modules.decoder.crf import CRF

class LstmCrfTagger(Model):
    def __init__(self, label_nums, vocab_size, input_dim,
                hidden_size, num_layers, use_crf=True,
                bidirection=True, batch_first=True, device=None,
                dropout=0.0, averge_batch_loss=True, **kwargs):
        """
        ref: Neural Architectures for Named Entity Recognition
        模型结构是word_embedding + bilstm + crf

        :params 
        """
        super(LstmCrfTagger, self).__init__(input_dim, vocab_size, **kwargs)

        self.encoder = LstmEncoderLayer(input_dim, hidden_size, num_layers, label_nums=label_nums+2,
                bidirectional=bidirection, batch_first=batch_first, dropout=dropout)

        self.averge_batch_loss = averge_batch_loss
        self.use_crf = use_crf
        if self.use_crf:
            self.decoder = CRF(label_nums, device)

    def forward(self, input, input_seq_length,
                mask=None, batch_label=None):

        embedding = self.embedding(input) #batch_size * seq_len * input_dim
        encoder_res = self.encoder(embedding, input_seq_length) #batch * seq_len * (hidden_dim*directions)
        batch_size = encoder_res.size(0)
        seq_len = encoder_res.size(1)

        if self.use_crf:
            _, tag_seq = self.decoder._viterbi_decode(encoder_res, mask)
            if batch_label is not None:
                total_loss = self.decoder.neg_log_likelihood_loss(encoder_res, mask, batch_label)
            
        else:
            outs = encoder_res.view(batch_size * seq_len, -1)
            _, tag_seq  = torch.max(outs, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            tag_seq = mask.long() * tag_seq
            if batch_label is not None:
                loss_function = nn.NLLLoss(ignore_index=0, size_average=False)#mask的token不对最后的loss产生影响, 固定mask的label id为0
                score = F.log_softmax(outs, 1)
                total_loss = loss_function(score, batch_label.view(batch_size * seq_len))

        if batch_label is not None:
            if self.averge_batch_loss:
                total_loss = total_loss / batch_size
            return {"loss":total_loss, "logits": tag_seq}
        else:
            return {"logits": tag_seq}

    def predict(self, input, input_seq_length, mask=None):
        input = input.unsqueeze(0)
        res = self.forward(input, input_seq_length, mask)
        return res["logits"]


        






