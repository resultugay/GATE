#!/bin/bash

python processDataRandom.py -data_dir '../../data/career/' -timelinessAttr "league_name, potential, player_positions, international_reputation" -checkpoint '../../data/career/pretrainedModel/' -gpu 0
