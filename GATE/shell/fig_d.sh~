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

echo -e 'dataset : '${data[${did}]}
epoch=10
lr=0.0001
batch_size=512
conf_sample_size=0.2
conf_threshold=0.55

echo -e "GATE"
mkdir results

echo -e "Method GATE"
resultFile='./result/fig_d_gate.txt'
> ${resultFile}
python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant gate --gpuOption ${gpu} >> ${resultFile} 


echo -e "Method Creator"
resultFile='./result/fig_d_creator.txt'
> ${resultFile}
python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creator --gpuOption ${gpu} >> ${resultFile} 


echo -e "Method Critic"
resultFile='./result/fig_d_critic.txt'
> ${resultFile}
python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant critic --gpuOption ${gpu} >> ${resultFile} 


echo -e "Method CreatorNC"
resultFile='./result/fig_d_creatornc.txt'
> ${resultFile}
python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatornc --gpuOption ${gpu} >> ${resultFile} 


echo -e "Method CreatorNA"
resultFile='./result/fig_d_creatorna.txt'
> ${resultFile}
python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatorna --gpuOption ${gpu} >> ${resultFile} 

echo -e "Method CreatorITR"
resultFile='./result/fig_d_creatoritr.txt'
> ${resultFile}
python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatoritr --gpuOption ${gpu} >> ${resultFile} 






