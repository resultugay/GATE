#!/bin/bash

data=(
'career'
'nba'
'person'
'comm'
)

did=3

cd ../

epoch=30
lr=0.0001
batch_size=512
conf_sample_size=0.2
conf_threshold=0.55

# Varying |\Sigma|

echo -e "GATE"
mkdir results

echo -e "Method GATE"

for gamma in 0.2 0.4 0.6 0.8
do
    resultFile='./result/fig_i_gate_'${data[${did}]}'_gamma'${gamma}'.txt'
    > ${resultFile}
    /root/anaconda3/bin/python main.py --creator Gate --data /data/data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant gate --gamma ${gamma} >> ${resultFile} 
done


echo -e "Method Critic"
for gamma in 0.2 0.4 0.6 0.8
do
    resultFile='./result/fig_i_critic_'${data[${did}]}'_gamma'${gamma}'.txt'
    > ${resultFile}
    /root/anaconda3/bin/python main.py --creator Gate --data /data/data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant critic --gamma ${gamma} >> ${resultFile} 
done


echo -e "Method Creator"
for gamma in 0.2 0.4 0.6 0.8
do
    resultFile='./result/fig_i_creator_'${data[${did}]}'_gamma'${gamma}'.txt'
    > ${resultFile}
    /root/anaconda3/bin/python main.py --creator Gate --data /data/data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creator --gamma ${gamma} >> ${resultFile} 

done


echo -e "Method CreatorNC"
for gamma in 0.2 0.4 0.6 0.8
do
    resultFile='./result/fig_i_creatornc_'${data[${did}]}'_gamma'${gamma}'.txt'
    > ${resultFile}
    /root/anaconda3/bin/python main.py --creator Gate --data /data/data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatornc --gamma ${gamma} >> ${resultFile} 

done

echo -e "Method CreatorNA"
for gamma in 0.2 0.4 0.6 0.8
do
    resultFile='./result/fig_i_creatorna_'${data[${did}]}'_gamma'${gamma}'.txt'
    > ${resultFile}
    /root/anaconda3/bin/python main.py --creator Gate --data /data/data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatorna --gamma ${gamma} >> ${resultFile} 

done






