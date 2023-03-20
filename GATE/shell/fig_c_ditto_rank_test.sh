#!/bin/bash

data=(
'career'
'nba'
'person'
'comm'
)

did=1
gpu=$1

cd ../baselines/ditto2_sf

#/data/qiu/tl_env2/bin/python baselines/ditto2_sf/ditto_rank_timeliness.py ${data[${did}]} 0 >> ${resultFile}
/data/qiu/tl_env2/bin/python ditto_rank_timeliness.py --dataset_type ${data[${did}]} --gpuOption ${gpu} --gamma 1.0 #>> ${resultFile}

