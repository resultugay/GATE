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

# Varying |\Sigma|

echo -e "GATE"
mkdir results

echo -e "Method GATE"

for cc in 1 2 3 4 
do
    resultFile='./result/fig_g_gate_'${data[${did}]}'_CC'${cc}'.txt'
    > ${resultFile}
    python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant gate --gpuOption ${gpu} --ccs 'CCs_'${cc}'.txt' >> ${resultFile} 
done


# echo -e "Method Critic"
# for cc in 1 2 3 4
# do
#     resultFile='./result/fig_g_critic_'${data[${did}]}'_CC'${cc}'.txt'
#     > ${resultFile}
#     python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant critic --ccs '../data/'${data[${did}]}'/CCs_'${cc}'.txt' >> ${resultFile} 
# done







