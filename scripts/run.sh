#!/bin/bash  
python train.py --model_name bert_multi_target --dataset restaurant --resplit 1 --device cuda:0  --aux 1
wait;
python train.py --model_name bert_multi_target --dataset restaurant --resplit 1 --device cuda:0  --adv 1
wait;

python train.py --model_name bert_multi_target --dataset restaurant --resplit 1 --device cuda:0  --aux 1 --adv 1 
wait;

python train.py --model_name bert_multi_target --dataset restaurant --resplit 1 --device cuda:0  --aux 1         --aug synonyms 
wait;

python train.py --model_name bert_multi_target --dataset restaurant --resplit 1 --device cuda:0          --adv 1 --aug synonyms 
wait;

python train.py --model_name bert_multi_target --dataset restaurant --resplit 1 --device cuda:0  --aux 1 --adv 1 --aug synonyms 
wait;

python train.py --model_name bert_multi_target --dataset restaurant --resplit 1 --device cuda:0  --aux 1 --adv 1 --aug synonyms 
wait;

python train.py --model_name bert_multi_target --dataset restaurant --resplit 1 --device cuda:0  --aux 1         --aug synonyms --chg 1
wait;

python train.py --model_name bert_multi_target --dataset restaurant --resplit 1 --device cuda:0          --adv 1 --aug synonyms --chg 1
wait;

python train.py --model_name bert_multi_target --dataset restaurant --resplit 1 --device cuda:0  --aux 1 --adv 1 --aug synonyms --chg 1
wait;

python train.py --model_name bert_multi_target --dataset restaurant --resplit 1 --device cuda:0  --aux 1 --adv 1 --aug synonyms --chg 1
wait;







# for((i=1;i<=10;i++));  
# do   
#     python train.py --model_name bert_multi_target --dataset restaurant --device cuda:0;
#     wait;
# done  