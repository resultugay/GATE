#!/bin/bash

/root/anaconda3/bin/python discovery.py --data_dir /data/data/nba_/data.csv --min_freq 0.1 --max_freq 0.8 --relation nba --timelinessAttrs "team_abbreviation, player_weight, college" --dataProcessedFile /data/data/nba_/training_processed.pkl
