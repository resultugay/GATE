#!/bin/bash

data=(
'career'
'nba'
'person'
'comm_'
)

did=3
gpu=$1

cd ../

resultFile='./result/fig_d_rank_bert.txt'
~/anaconda3/bin/python baselines/rank_bert.py ${data[${did}]} ${gpu} 1.0 >> ${resultFile}
#~/anaconda3/bin/python baselines/rank_bert.py ${data[${did}]} '' 1.0 >> ${resultFile}

