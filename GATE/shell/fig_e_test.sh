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
lr=0.00005
batch_size=512
conf_sample_size=1.0
#conf_threshold=0.55
conf_threshold=0.53

echo -e "GATE"
mkdir results

echo -e "Method GATE"
python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant gate --gpuOption ${gpu} --maxMLData 1000 #>> 'result_person_test.txt' 


# echo -e "Method Creator"
# resultFile='./result/fig_e_creator.txt'
# > ${resultFile}
# python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creator --gpuOption ${gpu} >> ${resultFile} 


# echo -e "Method Critic"
# resultFile='./result/fig_e_critic.txt'
# > ${resultFile}
# python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant critic --gpuOption ${gpu} >> ${resultFile} 


# echo -e "Method CreatorNC"
# resultFile='./result/fig_e_creatornc.txt'
# > ${resultFile}
# python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatornc --gpuOption ${gpu} >> ${resultFile} 


# echo -e "Method CreatorNA"
# resultFile='./result/fig_e_creatorna.txt'
# > ${resultFile}
# python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatorna --gpuOption ${gpu} >> ${resultFile} 

# echo -e "Method CreatorITR"
# resultFile='./result/fig_e_creatoritr.txt'
# > ${resultFile}
# python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatoritr --gpuOption ${gpu} >> ${resultFile} 






