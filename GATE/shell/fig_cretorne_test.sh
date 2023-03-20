#!/bin/bash

data=(
'career'
'nba'
'person'
'comm_'
)

gpu=$1

cd ../

echo -e 'dataset : '${data[${did}]}
epoch=30
lr=0.0001
batch_size=512
conf_sample_size=0.2
conf_threshold=0.55


# Fig a
did=0
echo -e "Method CreatorNE"
resultFile='./result/fig_a_creatorne.txt'
python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatorne --gpuOption ${gpu} #>> ${resultFile} 




