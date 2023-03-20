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

echo -e 'dataset : '${data[${did}]}
epoch=60
lr=0.0001
batch_size=512
conf_sample_size=0.5
conf_threshold=0.55

echo -e "GATE"
mkdir results

echo -e "Method GATE"

for conf in 0.52 #0.51 0.52 0.53 0.54 # 0.5 0.6 0.65 0.7 0.75 
do
    resultFile='./result/fig_l_gate_person_conf='${conf}'.txt'
    #> ${resultFile}
    python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf} --variant gate --gpuOption ${gpu}  #>> ${resultFile} 
done

for conf in 0.5 0.6 0.65 0.7 0.75 
do
    resultFile='./result/fig_l_creatoritr_person_conf='${conf}'.txt'
    #> ${resultFile}
    python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf} --variant creatoritr --gpuOption ${gpu}  >> ${resultFile} 
done








