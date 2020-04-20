# -*- coding: utf-8 -*-
# file: train.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy
import numpy as np






from pytorch_transformers import BertModel,BertForTokenClassification,BertConfig
# from transformers import BertModel,BertForTokenClassification,BertConfig

# from models.knowledge_bert import BertForTokenClassification

from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset

from models import LSTM, IAN, MemNet, RAM, TD_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, LCF_BERT
from models.aen import CrossEntropyLoss_LSR, AEN_BERT
from models.bert_spc import BERT_SPC
from models.bert_raw import BERT_RAW
from models.bert_label import BERT_LABEL
from models.bert_aspect import BERT_ASPECT
from models.bert_target import BERT_TARGET
from models.bert_multi_target import BERT_MULTI_TARGET
from models.bert_kg import BERT_KG
from models.bert_compete import BERT_COMPETE


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from torch.autograd import Variable, grad


reg_list=['bert_compete','bert_multi_target']
last_model_path=None



class Instructor:
    def __init__(self, opt):
        self.opt = opt

        if 'bert' in opt.model_name:
            if opt.model_name == 'bert_kg':
                tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
                bert = BertForTokenClassification.from_pretrained('ernie_base')
                self.model = opt.model_class(bert, opt).to(opt.device)
                self.model.to(opt.device)
            elif opt.model_name == 'bert_spc' :
                tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
                config = BertConfig.from_pretrained(opt.pretrained_bert_name, output_attentions=True)
                bert = BertModel.from_pretrained(opt.pretrained_bert_name,config=config)
                self.model = opt.model_class(bert, opt).to(opt.device)
                # self.model.load_state_dict(torch.load('./state_dict/bert_multi_target_val_acc0.7714'))  
            elif opt.model_name == 'bert_label' :
                tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
                config = BertConfig.from_pretrained(opt.pretrained_bert_name, output_attentions=True)
                bert = BertModel.from_pretrained(opt.pretrained_bert_name,config=config)
                self.model = opt.model_class(bert, opt).to(opt.device)
            elif opt.model_name == 'bert_compete' :
                tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
                config = BertConfig.from_pretrained(opt.pretrained_bert_name, output_attentions=True)
                bert = BertModel.from_pretrained(opt.pretrained_bert_name,config=config)

                num_added_tokens = tokenizer.add_tokens(['[aspect_b]','[aspect_e]'])
                bert.resize_token_embeddings(len(tokenizer.tokenizer))
                self.model = opt.model_class(bert, opt).to(opt.device)
            else:

                # bert_mulit_target
                tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
                config = BertConfig.from_pretrained(opt.pretrained_bert_name, output_attentions=True)
                bert = BertModel.from_pretrained(opt.pretrained_bert_name,config=config)
                if opt.domain=='pt':
                    bert = BertModel.from_pretrained('./bert_models/pt_bert-base-uncased_amazon_yelp')
                if opt.domain=='joint':
                    bert = BertModel.from_pretrained('./bert_models/laptops_and_restaurants_2mio_ep15')
                if opt.domain=='res':
                    bert = BertModel.from_pretrained('./bert_models/restaurants_10mio_ep3')  
                if opt.domain=='laptop':
                    bert = BertModel.from_pretrained('./bert_models/laptops_1mio_ep30')  


                num_added_tokens = tokenizer.add_tokens(['[target_b]','[target_e]'])
                num_added_tokens = tokenizer.add_tokens(['[aspect_b]','[aspect_e]'])
                for i in range(20):
                    b='['+str(i)+'b]'
                    e='['+str(i)+'e]'
                    num_added_tokens = tokenizer.add_tokens([b,e])           
                bert.resize_token_embeddings(len(tokenizer.tokenizer))
                self.model = opt.model_class(bert, opt).to(opt.device)
                # self.model.load_state_dict(torch.load('./state_dict/state_dict/bert_multi_target_restaurant_doamin-res_can0_adv0_aux1.0_val_acc0.8688'))  


        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer,'train',opt)
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer,'test',opt)

        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        
        # if opt.load_mode == 1:
            # self.model.load_state_dict(torch.load('/home/nus/temp/ABSA-PyTorch/state_dict/bert_spc_twitter_val_acc0.7384'))
        # find the highese
        # model.load_state_dict(torch.load(PATH))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _l2_normalize(self,d):
        if isinstance(d, Variable):
            d = d.data.cpu().numpy()
        elif isinstance(d, torch.FloatTensor) or isinstance(d, torch.cuda.FloatTensor):
            d = d.cpu().numpy()
        d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape((-1, 1, 1)) + 1e-16)
        return torch.from_numpy(d)
    
    def _loss_adv(self,loss,emb,criterion,inputs,targets,p_mult):
        emb_grad = grad(loss, emb, retain_graph=True)
        p_adv = torch.FloatTensor(p_mult * self._l2_normalize(emb_grad[0].data))
        p_adv=p_adv.cuda(non_blocking=False)
        p_adv = Variable(p_adv)

        outputs,reg,bert_word_output = self.model(inputs,p_adv)
        adv_loss = criterion(outputs, targets)
        # loss += adv_loss
        return adv_loss

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        global_step = 0
        last_model_path = None
        path=None
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                if self.opt.model_name=='bert_multi_target':
                    targets = sample_batched['polarity'].to(self.opt.device)
                    # print(targets.shape)
                else:
                    targets = sample_batched['polarity'].to(self.opt.device)
                if self.opt.model_name in reg_list:
                    aux_cls_logeits,outputs,reg_can_loss,reg_aux_loss,bert_word_output = self.model(inputs,None)
                else:
                    outputs=self.model(inputs)
                    reg_loss=0
                
                # print('outputs',outputs.shape)
                # print('targets',targets.shape)

                loss_1 = criterion(outputs, targets)
                loss_2 = reg_can_loss
                loss_3 = reg_aux_loss
                weighted_loss_2 =   loss_2 * self.opt.can
                weighted_loss_3 =   loss_3 * self.opt.aux
                loss= 0*loss_1 + weighted_loss_2 + weighted_loss_3


                if self.opt.adv > 0:
                    # print(inputs.shape)
                    loss_adv = self._loss_adv(loss,bert_word_output,criterion,inputs,targets,p_mult=self.opt.adv)
                    loss+=loss_adv
                else:
                    loss_adv=0
                loss.backward()
                optimizer.step()

                # n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_correct += (torch.argmax(aux_cls_logeits, -1) == 4*t_targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss_total: {:.4f}, acc: {:.4f},loss_main: {:.4f},loss_reg2: {:.4f},loss_adv: {:.4f},loss_reg3 {:.4f}'.format(train_loss, train_acc,loss_1,weighted_loss_2,loss_adv,weighted_loss_3))

            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                model_path = 'state_dict/{0}_{1}_doamin-{2}_can{3}_adv{4}_aux{5}_val_acc{6}'.format(self.opt.model_name,self.opt.dataset,self.opt.domain,self.opt.can,self.opt.adv,self.opt.aux,round(val_acc, 4))
                bert_path = 'state_dict/{0}_{1}_doamin-{2}_can{3}_adv{4}_aux{5}_val_acc{6}_bert'.format(self.opt.model_name, self.opt.dataset,self.opt.domain,self.opt.can,self.opt.adv,self.opt.aux,round(val_acc, 4))
                

                if last_model_path!=None:
                    os.remove(last_model_path)
                    os.remove(last_bert_path)
                last_model_path=model_path
                last_bert_path=bert_path
                torch.save(self.model.state_dict(), model_path)
                torch.save(self.model.bert.state_dict(), bert_path)
                logger.info('>> saved: {}'.format(model_path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1

        return model_path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                # print('t_sample_batched',t_sample_batched)
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(self.opt.device)
                if self.opt.model_name in reg_list:
                    # t_outputs,reg_less,emb = self.model(t_inputs,None)
                    aux_cls_logeits,t_outputs,reg_can_loss,reg_aux_loss,bert_word_output = self.model(t_inputs,None)
                else:
                    t_outputs= self.model(t_inputs)

                if 1:
                    n_correct += (torch.argmax(aux_cls_logeits, -1) == 4*t_targets).sum().item()
                    n_total += len(t_outputs)
                else:
                    n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                    n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return acc, f1

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        # self._reset_params()
        # self.model.load_state_dict(torch.load('./state_dict/bert_spc_restaurant_val_acc0.7893'))
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        # self.model.load_state_dict(torch.load('state_dict/bert_spc_restaurant_val_acc0.6491'))
        self.model.eval()
        test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_spc', type=str)
    parser.add_argument('--dataset', default='laptop', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=10, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=128, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    # parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default='cuda:1', type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=None, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--load_mode', default=0, type=int, help='load existed model')

    parser.add_argument('--can', default=0, type=float, help='using tfm')
    parser.add_argument('--adv', default=0, type=float, help='using adv training')
    parser.add_argument('--aux', default=0, type=float, help='using aux training')

    parser.add_argument('--domain', default=0, type=str, help='using domain bert')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int, help='semantic-relative-distance, see the paper of LCF-BERT model')
    opt = parser.parse_args()
    torch.cuda.set_device(opt.device)
    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_classes = {
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
        'aoa': AOA,
        'mgan': MGAN,
        'bert_spc': BERT_SPC,
        'aen_bert': AEN_BERT,
        'lcf_bert': LCF_BERT,
        'bert_raw': BERT_RAW,
        'bert_label': BERT_LABEL,
        'bert_aspect': BERT_ASPECT,
        'bert_kg': BERT_KG,
        'bert_compete': BERT_COMPETE,
        'bert_multi_target':BERT_MULTI_TARGET,
        'bert_target':BERT_TARGET,
        # default hyper-parameters for LCF-BERT model is as follws:
        # lr: 2e-5
        # l2: 1e-5
        # batch size: 16
        # num epochs: 5
    }
    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': './datasets/semeval14/Restaurants_Train.xml.seg',
            'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': './datasets/semeval14/Laptops_Train.xml.seg',
            'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
        }
    }
    input_colses = {
        'lstm': ['text_raw_indices'],
        'td_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'atae_lstm': ['text_raw_indices', 'aspect_indices'],
        'ian': ['text_raw_indices', 'aspect_indices'],
        'memnet': ['text_raw_without_aspect_indices', 'aspect_indices'],
        'ram': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'cabasc': ['text_raw_indices', 'aspect_indices', 'text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'tnet_lf': ['text_raw_indices', 'aspect_indices', 'aspect_in_text'],
        'aoa': ['text_raw_indices', 'aspect_indices'],
        'mgan': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'bert_spc': ['text_bert_indices', 'bert_segments_ids'],
        'aen_bert': ['text_raw_bert_indices', 'aspect_bert_indices'],
        'lcf_bert': ['text_bert_indices', 'bert_segments_ids', 'text_raw_bert_indices', 'aspect_bert_indices'],
        'bert_raw': ['text_raw_bert_indices', 'bert_raw_segments_ids'],
        'bert_label': ['text_raw_bert_indices', 'bert_segments_ids','polarity'],
        

        # 'bert_aspect': ['bert_aspect_indices','bert_aspect_segments_ids','aspect_in_text','aspect_len'],
        'bert_aspect': ['text_raw_bert_indices','bert_raw_segments_ids','aspect_in_text','aspect_len'],
        
        'bert_target': ['text_target_indices', 'text_target_segments_ids','target_begin'],
        'bert_multi_target': ['multi_target_indices','multi_target_segments_ids','target_pos','poss','polarity_list','polarity'],

        'bert_kg': ['text_bert_indices', 'bert_segments_ids','input_mask'],
        'bert_compete':['bert_compete_cls_pos','bert_compete_indices','bert_compete_segments_ids','bert_compete_cls_poss']
        # 'bert_kg': ['text_bert_indices', 'bert_segments_ids','input_mask'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = './log/{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
