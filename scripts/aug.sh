#!/bin/bash  
  
python train.py --model_name bert_multi_target --dataset restaurant  --device cuda:0 --resplit 0 --valset_ratio 0.05 --aug synonyms
wait
python train.py --model_name bert_multi_target --dataset restaurant  --device cuda:0 --resplit 1 --valset_ratio 0.05 --aug synonyms --chg 1
