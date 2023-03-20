#!/bin/bash

data=(
'career'
'nba'
'person'
'comm'
)

did=2
gpu=$1

cd ../

for gamma in 0.2 0.4 0.6 0.8 1.0 #0.6 
do
    resultFile='./result/fig_e_rank_bert_gamma='${gamma}'.txt'
    ~/anaconda3/bin/python baselines/rank_bert.py ${data[${did}]} ${gpu} ${gamma} >> ${resultFile}
done
