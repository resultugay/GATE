#!/bin/bash

# sigmod second round ddl version
#/root/anaconda3/bin/python processDataComm.py -data_dir '/data/data/comm/' -timelinessAttr "name, address, owner"

python processDataRandom.py -data_dir '../../data/comm/' -timelinessAttr "name, address, owner" -checkpoint '../../data/comm/pretrainedModel/' -gpu 2
