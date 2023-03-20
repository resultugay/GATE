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

epoch=30
lr=0.0001
batch_size=512
conf_sample_size=0.2
conf_threshold=0.55

mkdir results

echo -e "Method CreatorITR"
resultFile='./result/fig_d_creatoritr_comm_.txt'
> ${resultFile}
python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatoritr --gpuOption ${gpu} >> ${resultFile} 








