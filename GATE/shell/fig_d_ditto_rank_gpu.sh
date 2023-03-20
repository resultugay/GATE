#!/bin/bash

data=(
'career'
'nba'
'person'
'comm_'
)

did=3
gpu=$1

cd ../baselines/ditto2_sf

resultFile='../../result/fig_d_ditto_rank_gpu.txt'
#/data/qiu/tl_env2/bin/python baselines/ditto2_sf/ditto_rank_timeliness.py ${data[${did}]} 0 >> ${resultFile}
# has to set max_len as 256 due to out of cuda memory 
/data/qiu/tl_env2/bin/python ditto_rank_timeliness.py --dataset_type ${data[${did}]} --gpuOption ${gpu} --gamma 1.0 --max_len 256 >> ${resultFile}
#/data/qiu/tl_env2/bin/python ditto_rank_timeliness.py --dataset_type ${data[${did}]} --gamma 1.0 >> ${resultFile}

