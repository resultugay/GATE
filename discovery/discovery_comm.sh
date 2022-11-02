#!/bin/bash

/root/anaconda3/bin/python discovery.py --data_dir /data/data/comm/data.csv --min_freq 0.1 --max_freq 0.8 --relation comm --timelinessAttrs "name, address, owner" --dataProcessedFile /data/data/comm/training_processed.pkl
