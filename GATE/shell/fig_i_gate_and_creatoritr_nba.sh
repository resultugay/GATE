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

epoch=30
lr=0.0001
batch_size=512
conf_sample_size=0.2
conf_threshold=0.5

# Varying |\Sigma|

echo -e "GATE"
mkdir results

echo -e "Method GATE"

for gamma in 0.2 0.4 0.6 0.8
do
    resultFile='./result/fig_i_gate_'${data[${did}]}'_gamma'${gamma}'_conf=0.5.txt'
    > ${resultFile}
    python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant gate --gpuOption ${gpu} --gamma ${gamma} >> ${resultFile} 
done

echo -e "Method CreatorITR"
for gamma in 0.2 0.4 0.6 0.8
do
    resultFile='./result/fig_i_creatoritr_'${data[${did}]}'_gamma'${gamma}'_conf=0.5.txt'
    > ${resultFile}
    python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatoritr --gpuOption ${gpu} --gamma ${gamma} >> ${resultFile} 

done





