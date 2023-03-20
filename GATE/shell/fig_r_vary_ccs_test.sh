#!/bin/bash

data=(
'career'
'nba'
'person'
'comm_'
)

did=2
gpu=$1

cd ../

echo -e 'dataset : '${data[${did}]}
epoch=60
lr=0.0001
batch_size=512
conf_sample_size=0.2
conf_threshold=0.55

echo -e "GATE"
mkdir results

echo -e "Method GATENC and GATE"

#for dt in 0.2 0.4 0.6 0.8
for cc in 1 # 2 3 4 
do
    resultFile='./result/fig_r_gate_CCs='${cc}'.txt'
    python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant gate --gpuOption ${gpu} --ccs '../data/'${data[${did}]}'/CCs_'${cc}'.txt' #>> ${resultFile} 

done




