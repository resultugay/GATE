
#!/bin/bash

data=(
'career'
'nba'
'person'
'comm_'
)

did=$1
gpu=$2

python ../pretrain/bert_pretrain.py -data_dir ../../data/${data[${did}]} -gpu ${gpu} -saved_model ../../data/${data[${did}]}/pretrainedModel/
