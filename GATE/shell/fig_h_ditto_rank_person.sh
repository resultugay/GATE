#!/bin/bash

data=(
'career'
'nba'
'person'
'comm'
)

did=2
gpu=$1

cd ../baselines/ditto2_sf

for gamma in 0.2 0.4 0.6 0.8 #0.6
do
    resultFile='../../result/fig_e_ditto_rank_gamma='${gamma}'.txt'
    #/data/qiu/tl_env2/bin/python baselines/ditto2_sf/ditto_rank_timeliness.py ${data[${did}]} 0 >> ${resultFile}
    /data/qiu/tl_env2/bin/python ditto_rank_timeliness.py --dataset_type ${data[${did}]} --gpuOption ${gpu} --gamma ${gamma} >> ${resultFile}
done
