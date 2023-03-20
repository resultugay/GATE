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
echo -e "Method CreatorNC"
resultFile='./result/fig_a_creatornc.txt'
#> ${resultFile}
#python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatornc --gpuOption ${gpu} >> ${resultFile} 


echo -e "Method CreatorNE"
resultFile='./result/fig_a_creatorne.txt'
#> ${resultFile}
python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatorne --gpuOption ${gpu} >> ${resultFile} 


# Fig c
did=1
echo -e "Method CreatorNC"
resultFile='./result/fig_c_creatornc.txt'
#> ${resultFile}
#python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatornc --gpuOption ${gpu} >> ${resultFile} 


echo -e "Method CreatorNE"
resultFile='./result/fig_c_creatorne.txt'
#> ${resultFile}
python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatorne --gpuOption ${gpu} >> ${resultFile} 


# Fig d
did=3
echo -e "Method CreatorNC"
resultFile='./result/fig_d_creatornc.txt'
#> ${resultFile}
#python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatornc --gpuOption ${gpu} >> ${resultFile} 


echo -e "Method CreatorNE"
resultFile='./result/fig_d_creatorne.txt'
#> ${resultFile}
python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatorne --gpuOption ${gpu} >> ${resultFile} 


# Fig e
epoch=60

did=2
echo -e "Method CreatorNC"
resultFile='./result/fig_e_creatornc.txt'
#> ${resultFile}
#python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatornc --gpuOption ${gpu} >> ${resultFile} 


echo -e "Method CreatorNE"
resultFile='./result/fig_e_creatorne.txt'
#> ${resultFile}
python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatorne --gpuOption ${gpu} >> ${resultFile} 


# Fig h
epoch=60

did=2
echo -e "Method CreatorNC"
for gamma in 0.2 0.4 0.6 0.8
do
    resultFile='./result/fig_h_creatornc_'${data[${did}]}'_gamma'${gamma}'.txt'
    #> ${resultFile}
    #python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatornc --gpuOption ${gpu} --gamma ${gamma} >> ${resultFile} 

    resultFile='./result/fig_h_creatorne_'${data[${did}]}'_gamma'${gamma}'.txt'
    #> ${resultFile}
    python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatorne --gpuOption ${gpu} --gamma ${gamma} >> ${resultFile} 
done


# Fig i
epoch=30

did=1
echo -e "Method CreatorNC"
for gamma in 0.2 0.4 0.6 0.8
do
    resultFile='./result/fig_i_creatornc_'${data[${did}]}'_gamma'${gamma}'.txt'
    #> ${resultFile}
    #python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatornc --gpuOption ${gpu} --gamma ${gamma} >> ${resultFile} 

    resultFile='./result/fig_i_creatorne_'${data[${did}]}'_gamma'${gamma}'.txt'
    #> ${resultFile}
    python main.py --creator Gate --data ../data/${data[${did}]}'/' --epoch ${epoch} --lr ${lr} --batch_size ${batch_size} --high_conf_sample_ratio ${conf_sample_size} --conf_threshold ${conf_threshold} --variant creatorne --gpuOption ${gpu} --gamma ${gamma} >> ${resultFile} 
done



