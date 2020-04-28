# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
# from pytorch_transformers import BertTokenizer
from transformers import BertTokenizer

import nlpaug.augmenter.word as naw

aug = naw.SynonymAug()

def insert(original, new, pos):
# '''Inserts new inside original at pos.'''
    return original[:pos] + new + original[pos:]

def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


def pad(a,maxlen):
    B = np.pad(a, (0, maxlen - len(a)%maxlen), 'constant')
    return B


def pad_5(a,maxlen):
    B = np.pad(a, (5, maxlen - len(a)%maxlen), 'constant')
    return B


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)
    
    def add_tokens(self,params):
        self.tokenizer.add_tokens(params)

        


class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer,process,opt):
        

        all_data = []
        load = 1
        if int(opt.resplit)!=0 and load ==0:
            raise RuntimeError('pls use load to load the replit')

        if load:

            if fname.lower().find('train')!=-1:
                process='train'
            if fname.lower().find('test')!=-1:
                process='test'
            if fname.lower().find('valid')!=-1:
                process='valid'
            # if int(opt.aug)==1:
            #     all_data=np.load('./datasets/aug/{}-{}.npy'.format(process, opt.dataset),allow_pickle=True).tolist()
            # else:
            if int(opt.resplit)==1:
                print('./datasets/resplit/{}-{}.npy'.format(process, opt.dataset))
                all_data=np.load('./datasets/resplit/{}-{}.npy'.format(process, opt.dataset),allow_pickle=True).tolist()
            if int(opt.resplit)==2:
                print('./datasets/resplit/{}-{}-lw.npy'.format(process, opt.dataset))
                all_data=np.load('./datasets/resplit/{}-{}-lw.npy'.format(process, opt.dataset),allow_pickle=True).tolist()
            if int(opt.resplit)==3:
                print('./datasets/remove/{}-{}.npy'.format(process, opt.dataset))
                all_data=np.load('./datasets/remove/{}-{}.npy'.format(process, opt.dataset),allow_pickle=True).tolist()
            if int(opt.resplit)==0:
                all_data=np.load('./datasets/processed/{}-{}.npy'.format(process, opt.dataset),allow_pickle=True).tolist()

            self.data = all_data
        else:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            
            for i in range(0, len(lines), 3):

                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                polarity = lines[i + 2].strip()

                text_raw="[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]"
                text_spc='[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]"
                text_target='[CLS] ' + text_left + ' [aspect_b] '+aspect + ' [aspect_e] '+ text_right + ' [SEP] '

                text_without_cls=text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]"
                
                
                text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)


                text_target_indices = tokenizer.text_to_sequence(text_target)
                text_target_segments_ids=np.asarray([0] * (np.sum(text_target_indices != 0)))
                text_target_segments_ids = pad_and_truncate(text_target_segments_ids, tokenizer.max_seq_len)
                

                text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
                text_left_indices = tokenizer.text_to_sequence(text_left)
                text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
                text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
                text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right, reverse=True)
                aspect_indices = tokenizer.text_to_sequence(aspect)
                left_context_len = np.sum(text_left_indices != 0)
                aspect_len = np.sum(aspect_indices != 0)
                aspect_pos = left_context_len+1
                target_begin=left_context_len+1
                aspect_in_text = torch.tensor([left_context_len.item(), (left_context_len + aspect_len - 1).item()])
                # aspect_range = torch.LongTensor(range(left_context_len.item()+1, (left_context_len + aspect_len).item()+1))# plus [cls]
                polarity = int(polarity) + 1

                text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
                # text_bert_indices = tokenizer.text_to_sequence('[CLS] '+ text_left + " " + aspect + " " + text_right + ' [SEP] '+ aspect + " [SEP] ")

                bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0)+2) + [1] * (aspect_len + 1))
                # bert_segments_ids = np.asarray([1] * (aspect_len + 1)+[0] * (np.sum(text_raw_indices != 0) + 2))
                bert_raw_segments_ids=np.asarray([0] * (np.sum(text_raw_indices != 0)+2))
                bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)
                bert_raw_segments_ids = pad_and_truncate(bert_raw_segments_ids, tokenizer.max_seq_len)
                text_raw_bert_indices = tokenizer.text_to_sequence("[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
                aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")
                input_mask=torch.tensor([1]*len(text_bert_indices))
                # print(aspect_indices)

                isaug=torch.tensor(0)
                data = {
                    'text_target_indices':text_target_indices,
                    'text_target_segments_ids':text_target_segments_ids,
                    'text_left':text_left,
                    'text_right':text_right,
                    'aspect_pos':aspect_pos,
                    'aspect_len':aspect_len,
                    'target_begin':target_begin,
                    'text_raw': text_raw,
                    'text_spc': text_spc,
                    'text_without_cls': text_without_cls,
                    'text_aspect':aspect,
                    'left_context_len': left_context_len,
                    'text_bert_indices': text_bert_indices,
                    'bert_segments_ids': bert_segments_ids,
                    'text_raw_bert_indices': text_raw_bert_indices,
                    'aspect_bert_indices': aspect_bert_indices,
                    'text_raw_indices': text_raw_indices,
                    'bert_raw_segments_ids':bert_raw_segments_ids,
                    'text_raw_without_aspect_indices': text_raw_without_aspect_indices,
                    'text_left_indices': text_left_indices,
                    'text_left_with_aspect_indices': text_left_with_aspect_indices,
                    'text_right_indices': text_right_indices,
                    'text_right_with_aspect_indices': text_right_with_aspect_indices,
                    'aspect_indices': aspect_indices,
                    'aspect_in_text': aspect_in_text,
                    'polarity': polarity,
                    'input_mask':input_mask,
                    'isaug':isaug,
                }

                all_data.append(data)

        
        l=len(all_data)
        if opt.aug==1:
            idx=0       
            while idx in range(l):
                print(idx)
                
                data=all_data[idx]
                text_left=data['text_left']
                text_right=data['text_right']

                aspect=data['text_aspect']
                polarity=data['polarity']




                # isaug=1
                isaug=torch.tensor(1)

                ori_aspect=aspect
                augmented_aspect = aug.augment(aspect)
                aspect=augmented_aspect
                print(ori_aspect,'---->',aspect)

                text_raw="[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]"
                text_spc='[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]"
                text_target='[CLS] ' + text_left + ' [aspect_b] '+aspect + ' [aspect_e] '+ text_right + ' [SEP] '

                text_without_cls=text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]"
                
                
                text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)


                text_target_indices = tokenizer.text_to_sequence(text_target)
                text_target_segments_ids=np.asarray([0] * (np.sum(text_target_indices != 0)))
                text_target_segments_ids = pad_and_truncate(text_target_segments_ids, tokenizer.max_seq_len)
                

                text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
                text_left_indices = tokenizer.text_to_sequence(text_left)
                text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
                text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
                text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right, reverse=True)
                aspect_indices = tokenizer.text_to_sequence(aspect)
                left_context_len = np.sum(text_left_indices != 0)
                aspect_len = np.sum(aspect_indices != 0)
                aspect_pos = left_context_len+1
                target_begin=left_context_len+1
                aspect_in_text = torch.tensor([left_context_len.item(), (left_context_len + aspect_len - 1).item()])
                # aspect_range = torch.LongTensor(range(left_context_len.item()+1, (left_context_len + aspect_len).item()+1))# plus [cls]
                # polarity = int(polarity) + 1

                text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
                # text_bert_indices = tokenizer.text_to_sequence('[CLS] '+ text_left + " " + aspect + " " + text_right + ' [SEP] '+ aspect + " [SEP] ")

                bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0)+2) + [1] * (aspect_len + 1))
                # bert_segments_ids = np.asarray([1] * (aspect_len + 1)+[0] * (np.sum(text_raw_indices != 0) + 2))
                bert_raw_segments_ids=np.asarray([0] * (np.sum(text_raw_indices != 0)+2))
                bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)
                bert_raw_segments_ids = pad_and_truncate(bert_raw_segments_ids, tokenizer.max_seq_len)
                text_raw_bert_indices = tokenizer.text_to_sequence("[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
                aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")
                input_mask=torch.tensor([1]*len(text_bert_indices))
                # print(aspect_indices)
                data = {
                    'text_target_indices':text_target_indices,
                    'text_target_segments_ids':text_target_segments_ids,
                    'text_left':text_left,
                    'text_right':text_right,
                    
                    'aspect_pos':aspect_pos,
                    'aspect_len':aspect_len,
                    'target_begin':target_begin,
                    'text_raw': text_raw,
                    'text_spc': text_spc,
                    'text_without_cls': text_without_cls,
                    'text_aspect':aspect,
                    'left_context_len': left_context_len,
                    'text_bert_indices': text_bert_indices,
                    'bert_segments_ids': bert_segments_ids,
                    'text_raw_bert_indices': text_raw_bert_indices,
                    'aspect_bert_indices': aspect_bert_indices,
                    'text_raw_indices': text_raw_indices,
                    'bert_raw_segments_ids':bert_raw_segments_ids,
                    'text_raw_without_aspect_indices': text_raw_without_aspect_indices,
                    'text_left_indices': text_left_indices,
                    'text_left_with_aspect_indices': text_left_with_aspect_indices,
                    'text_right_indices': text_right_indices,
                    'text_right_with_aspect_indices': text_right_with_aspect_indices,
                    'aspect_indices': aspect_indices,
                    'aspect_in_text': aspect_in_text,
                    'polarity': polarity,
                    'input_mask':input_mask,
                    'isaug':isaug,
                }

                all_data.append(data)
                idx=idx+1


        target_b=tokenizer.text_to_sequence('[target_b]')[0]
        target_e=tokenizer.text_to_sequence('[target_e]')[0]

        # aspect_b=tokenizer.text_to_sequence('[aspect_b]')[0]
        # target_e=tokenizer.text_to_sequence('[aspect_e]')[0]
        aspect_b=[]
        aspect_e=[]

        aspect_b_tokens=[]
        aspect_e_tokens=[]

        for i in range(20):
            b='['+str(i)+'b]'
            e='['+str(i)+'e]'
            aspect_b_tokens.append(b)
            aspect_e_tokens.append(e)

            aspect_b.append(tokenizer.text_to_sequence(b)[0])
            aspect_e.append(tokenizer.text_to_sequence(e)[0])



        idx=0       
        while idx in range(len(all_data)):
            # print(idx)
            

            data=all_data[idx]
            text_raw=data['text_raw']
            flag = True
            count=0
            while flag:
                count=count+1
                if idx+count not in range(len(all_data)):
                    break
                text_raw_next=all_data[idx+count]['text_raw']
                if (text_raw_next!=text_raw):
                    flag=False
            aspect_list=[]
            polarity_list=[]
            for i in range(0,count):
                text_aspect=all_data[idx+i]['text_aspect']
                aspect_list.append(text_aspect)
                polarity_list.append(all_data[idx+i]['polarity']+1)
            for i in range(0,count):
                all_data[idx+i]['aspect_list']=aspect_list
                all_data[idx+i]['polarity_list']=polarity_list
            idx=idx+count
        
        for i in range(len(all_data)):
            all_data[i]['text_multi']=all_data[i]['text_raw']
            now=0
            # print(len(all_data[i]['aspect_list']),all_data[i]['aspect_list'])
            for aspect in all_data[i]['aspect_list']:
                aspect_len=len(aspect)
                text_multi=all_data[i]['text_multi']
                if aspect == all_data[i]['text_aspect']:
                    
                    aspect_b_now=aspect_b_tokens[now]
                    aspect_e_now=aspect_e_tokens[now]

                        
                    # text_multi=insert(text_multi,' [target_b] ',text_multi.find(aspect))
                    # text_multi=insert(text_multi,' [target_e] ',text_multi.find(aspect)+len(aspect))
                    text_multi=insert(text_multi,' '+aspect_b_now+' ',text_multi.find(aspect))
                    text_multi=insert(text_multi,' '+aspect_e_now+' ',text_multi.find(aspect)+len(aspect))
                    # main_target=aspect_b[now]
                    now=now+1
                    
                else:
                    aspect_b_now=aspect_b_tokens[now]
                    aspect_e_now=aspect_e_tokens[now]
                    text_multi=insert(text_multi,' '+aspect_b_now+' ',text_multi.find(aspect))
                    text_multi=insert(text_multi,' '+aspect_e_now+' ',text_multi.find(aspect)+len(aspect))
                    now=now+1
                    # text_multi=insert(text_multi,' [aspect_b] ',text_multi.find(aspect))
                    # text_multi=insert(text_multi,' [aspect_e] ',text_multi.find(aspect)+len(aspect))
                all_data[i]['text_multi']=text_multi
            

            multi_target_indices = tokenizer.text_to_sequence(all_data[i]['text_multi'])
            all_data[i]['multi_target_indices']=multi_target_indices
            
            now=0
            poss=[]
            for aspect in all_data[i]['aspect_list']:
                aspect_len=len(aspect)
                text_multi=all_data[i]['text_multi']
                if aspect == all_data[i]['text_aspect']:    
                    aspect_b_now=aspect_b[now]
                    aspect_e_now=aspect_e[now]
                    pos=np.argwhere(all_data[i]['multi_target_indices']==aspect_b_now)[0][0]
                    pos_end=np.argwhere(all_data[i]['multi_target_indices']==aspect_e_now)[0][0]
                    
                    
                    main_target_pos=pos
                    main_target_pos_end=pos_end
                    poss.append(pos)
                    now=now+1
                else:
                    aspect_b_now=aspect_b[now]
                    aspect_e_now=aspect_e[now]
                    pos=np.argwhere(all_data[i]['multi_target_indices']==aspect_b_now)[0][0]
                    poss.append(pos)
                    now=now+1
                    # text_multi=insert(text_multi,' [aspect_b] ',text_multi.find(aspect))
                    # text_multi=insert(text_multi,' [aspect_e] ',text_multi.find(aspect)+len(aspect))
                all_data[i]['text_multi']=text_multi
                
  
            multi_target_segments_ids=np.asarray([0] * (np.sum(multi_target_indices != 0)))
            multi_target_segments_ids = pad_and_truncate(multi_target_segments_ids, tokenizer.max_seq_len)
            all_data[i]['multi_target_segments_ids']=multi_target_segments_ids
            # pos=np.argwhere(all_data[i]['multi_target_indices']==target_b)[0]
            
            poss=pad(poss,maxlen=20)
            poss=torch.tensor(poss)
            # print(poss)
            polarity_list=all_data[i]['polarity_list']
            polarity_list=pad(polarity_list,maxlen=20)
            polarity_list=torch.tensor(polarity_list)
            # poss=np.argwhere(all_data[i]['multi_target_indices']==target_b)[0]

            # aspect_poss=np.argwhere(all_data[i]['multi_target_indices']==target_b)[0][0]
            # print(pos,all_data[i]['multi_target_indices'])
            # print(main_target_pos)
            # print(main_target_pos_end)

            all_data[i]['target_pos']=main_target_pos
            all_data[i]['target_pos_end']=main_target_pos_end

            all_data[i]['poss']=poss
            all_data[i]['polarity_list']=polarity_list


        # out_name='./datasets/processed/{}-{}.txt'.format(process, opt.dataset)
        # out = open(out_name,'w+')
        # for i in range(len(all_data)):
        #     out.write(all_data[i]['text_multi'])
        #     out.write('\t')
        #     out.write(str(all_data[i]['polarity']))
        #     out.write('\n')
        # out.close()

        
        a=np.array(all_data)
        if opt.aug==1:
            np.save('./datasets/aug/{}-{}.npy'.format(process, opt.dataset),a)
        else:
            np.save('./datasets/processed/{}-{}.npy'.format(process, opt.dataset),a)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
