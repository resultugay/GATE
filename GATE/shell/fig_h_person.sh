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

# old bug version setting
# epoch=60
# lr=0.0001
# batch_size=512
# conf_sample_size=0.5
# conf_threshold=0.55

# new settings
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

#for gamma in 0.2 0.4 0.6 0.8 1.0
#do
#    resultFile='./result/fig_h_gate_fix_bug_'${data[${did}]}'_gamma'${gamma}'.txt'
#    > ${resultFile}
#    python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant gate --gpuOption ${gpu} --maxMLData 1000 --gamma ${gamma} >> ${resultFile}  
#done


# echo -e "Method Critic"
# for gamma in 0.2 0.4 0.6 0.8 1.0
# do
#     resultFile='./result/fig_h_critic_'${data[${did}]}'_gamma'${gamma}'.txt'
#     > ${resultFile}
#     python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant critic --gpuOption ${gpu} --gamma ${gamma} >> ${resultFile}  
# done


# echo -e "Method Creator"
# for gamma in 0.2 0.4 0.6 0.8 1.0
# do
#     resultFile='./result/fig_h_creator_'${data[${did}]}'_gamma'${gamma}'.txt'
#     > ${resultFile}
#     python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creator --gpuOption ${gpu} --gamma ${gamma} >> ${resultFile}  

# done


# echo -e "Method CreatorNC"
# for gamma in 0.2 0.4 0.6 0.8
# do
#     resultFile='./result/fig_h_creatornc_'${data[${did}]}'_gamma'${gamma}'.txt'
#     > ${resultFile}
#     python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatornc --gpuOption ${gpu} --gamma ${gamma} >> ${resultFile} 

# done

# echo -e "Method CreatorNA"
# for gamma in 0.2 0.4 0.6 0.8
# do
#     resultFile='./result/fig_h_creatorna_'${data[${did}]}'_gamma'${gamma}'.txt'
#     > ${resultFile}
#     python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatorna --gpuOption ${gpu} --gamma ${gamma} >> ${resultFile} 

# done

echo -e "Method CreatorITR"
for gamma in 1.0 # 0.2 0.4 0.6 0.8 1.0
do
    resultFile='./result/fig_h_creatoritr_fix_bug_'${data[${did}]}'_gamma'${gamma}'.txt'
    > ${resultFile}
    python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatoritr --gpuOption ${gpu} --gamma ${gamma} >> ${resultFile} 

done






