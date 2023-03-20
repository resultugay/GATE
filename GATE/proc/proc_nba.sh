#!/bin/bash

# sigmod second round revision
#/root/anaconda3/bin/python processData.py -data_dir '/data/data/nba/' -timelinessAttr "team_abbreviation, player_weight, college"

python processDataRandom.py -data_dir '../../data/nba/' -timelinessAttr "team_abbreviation, player_weight, pts" -checkpoint '../../data/nba/pretrainedModel/' -gpu 2
