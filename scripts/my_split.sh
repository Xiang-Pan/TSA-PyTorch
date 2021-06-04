#!/bin/bash  
  
for((i=1;i<=10;i++));  
do   
    python train.py --model_name bert_multi_target --dataset restaurant --device cuda:0;
    wait;
done  