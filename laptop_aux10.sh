#!/bin/bash  
  
for((i=1;i<=5;i++));  
do   
    python train.py --model_name bert_multi_target --dataset laptop --device cuda:0 --aux 10;
    wait;
done  