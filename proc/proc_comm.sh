#!/bin/bash

#/root/anaconda3/bin/python comm_data_proc.py '/data/data/comm/raw/comm.csv' '/data/data/comm/data.csv'

/root/anaconda3/bin/python processDataComm.py -data_dir '/data/data/comm/' -timelinessAttr "name, address, owner"
