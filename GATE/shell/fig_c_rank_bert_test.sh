#!/bin/bash

data=(
'career'
'nba'
'person'
'comm'
)

did=1
gpu=$1

cd ../

~/anaconda3/bin/python baselines/rank_bert.py ${data[${did}]} ${gpu} 1.0 #>> ${resultFile}

