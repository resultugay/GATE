#!/bin/bash

/root/anaconda3/bin/python discovery.py --data_dir /data/data/career/data.csv --min_freq 0.1 --max_freq 0.8 --relation career --timelinessAttrs "league_name, potential, player_positions, international_reputation" --dataProcessedFile /data/data/career/training_processed.pkl
