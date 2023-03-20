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

# bug version setting
# epoch=60
# lr=0.0001
# batch_size=512
# conf_sample_size=0.5
# conf_threshold=0.55

# new setting
epoch=60
lr=0.00005
batch_size=512
conf_sample_size=1.0
#conf_threshold=0.55
conf_threshold=0.53

# Varying |\Sigma|

echo -e "GATE"
mkdir results

echo -e "Method GATE"

for ratio in 0.2 0.4 0.6 0.8
do
   resultFile='./result/fig_j_gate_fix_bug_'${data[${did}]}'_ratio'${ratio}'.txt'
   > ${resultFile}
   python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant gate --gpuOption ${gpu} --maxMLData 1000 --entityRatio ${ratio} >> ${resultFile} 
done

for ratio in 0.2 0.4 0.6 0.8
do
    resultFile='./result/fig_j_creatoritr_fix_bug_'${data[${did}]}'_ratio'${ratio}'.txt'
    > ${resultFile}
    python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatoritr --gpuOption ${gpu} --entityRatio ${ratio} >> ${resultFile} 
done







