#!/bin/bash  
  
for((i=1;i<=5;i++));  
do   
    python train.py --model_name bert_spc --dataset laptop --device cuda:0;
    wait;
done  