#!/bin/bash

data=(
'career'
'nba'
'person'
'comm'
)

did=0


echo -e 'dataset : '${data[${did}]}
epoch=30
lr=0.0001
batch_size=512
conf_sample_size=0.2
conf_threshold=0.55

echo -e "GATE"
mkdir results

echo -e "Method GATE"
/root/anaconda3/bin/python main.py --creator Gate --data /data/data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant gate 







