#!/bin/bash  
python train.py --model_name bert_multi_target --dataset restaurant  --device cuda:1 --resplit 1 --aug 1
wait;
python train.py --model_name bert_multi_target --dataset restaurant  --device cuda:1 --resplit 1 --adv 1
wait;
python train.py --model_name bert_multi_target --dataset restaurant  --device cuda:1 --resplit 1 --aug 1 --adv 1
wait;
# python train.py --model_name bert_multi_target --dataset restaurant  --device cuda:0 --resplit 1 --aux 1 --adv 1 --adv_aux 1
# wait;