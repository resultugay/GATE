#!/bin/bash

data=(
'career'
'nba'
'person'
'comm'
)

did=0
gpu=$1

cd ../

resultFile='./result/fig_a_rank_bert.txt'
~/anaconda3/bin/python baselines/rank_bert.py ${data[${did}]} ${gpu} 1.0 >> ${resultFile}

