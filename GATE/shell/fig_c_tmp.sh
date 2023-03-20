#!/bin/bash

data=(
'career'
'nba'
'person'
'comm'
)

did=1

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
resultFile='./result/fig_c_gate.txt'
 > ${resultFile}
python main.py --creator Gate --data /tmp/yaoshuw/data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant gate >> ${resultFile} 


echo -e "Method Creator"
resultFile='./result/fig_c_creator.txt'
# > ${resultFile}
/root/anaconda3/bin/python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creator >> ${resultFile} 


echo -e "Method Critic"
resultFile='./result/fig_c_critic.txt'
# > ${resultFile}
/root/anaconda3/bin/python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant critic >> ${resultFile} 


echo -e "Method CreatorNC"
resultFile='./result/fig_c_creatornc.txt'
# > ${resultFile}
/root/anaconda3/bin/python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatornc >> ${resultFile} 


echo -e "Method CreatorNA"
resultFile='./result/fig_c_creatorna.txt'
# > ${resultFile}
/root/anaconda3/bin/python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatorna >> ${resultFile} 







