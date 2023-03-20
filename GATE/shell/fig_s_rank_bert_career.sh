#!/bin/bash

data=(
'career'
'nba'
'person'
'comm_'
)

did=0
gpu=$1

cd ../

for dt in 0.2 0.4 0.6 0.8 1.0 #0.6 
do
    resultFile='./result/fig_s_rank_bert_D_T='${dt}'.txt'
    ~/anaconda3/bin/python baselines/rank_bert.py ${data[${did}]} ${gpu} 1.0 ${dt} >> ${resultFile}
done
