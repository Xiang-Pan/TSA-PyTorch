#!/bin/bash  
  
for((i=1;i<=5;i++));  
do   
    python train.py --model_name bert_multi_target --dataset laptop --device cuda:1 --aux 0.01;
    wait;
done  