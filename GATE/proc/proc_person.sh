#!/bin/bash

# sigmod second round version
#/root/anaconda3/bin/python processDataRandom.py -data_dir '/data/data/person/' -timelinessAttr "LN, status, kids"

python processDataRandom.py -data_dir '../../data/person/' -timelinessAttr "LN, status, kids" -checkpoint '../../data/person/pretrainedModel/' -gpu 3

# python processDataRandom.py -data_dir '../../data/person/' -timelinessAttr "LN, status, kids" -gpu 3
