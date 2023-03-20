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

echo -e 'dataset : '${data[${did}]}
epoch=30
lr=0.0001
batch_size=512
conf_sample_size=0.2
conf_threshold=0.5

echo -e "GATE"
mkdir results

echo -e "Method GATE"
resultFile='./result/fig_c_gate_conf=0.5.txt'
 > ${resultFile}
python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant gate --gpuOption ${gpu}  >> ${resultFile} 

echo -e "Method CreatorITR"
resultFile='./result/fig_c_creatoritr_conf=0.5.txt'
> ${resultFile}
python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatoritr --gpuOption ${gpu} >> ${resultFile} 








