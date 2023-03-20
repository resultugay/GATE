import numpy as np
import pandas as pd
from collections import defaultdict
import itertools
import sys
import os
import pickle

meaninglessCols = ['id', 'entity_id', 'row_id', 'timestamp']

def clusterData(data):
    dataClusters = defaultdict(list)
    schemas = list(data.columns)
    data_np = data.values
    attrsMap = dict()
    for i, sc in enumerate(schemas):
        attrsMap[sc] = i
    for i, record in enumerate(data_np):
        dataClusters[record[attrsMap['entity_id']]].append(i)
    return dataClusters, schemas, attrsMap, np.array(data_np, 'str')

LIMITOR = '<--->'

def discoverCCs(dataClusters, schemas, attrsMap, data_np, timelinessAttrs):
    temporalOrdersDict = defaultdict(list) # ['<', '>']
    for eid, cluster in dataClusters.items():
        for sc_pair in itertools.permutations(cluster, 2):
            tid1, tid2 = sc_pair
            record1, record2 = data_np[tid1], data_np[tid2]
            for timelinessAttr in timelinessAttrs:
                k1, k2 = record1[attrsMap[timelinessAttr]], record2[attrsMap[timelinessAttr]]
                key_ = timelinessAttr + LIMITOR + k1 + LIMITOR + k2
                if key_ not in temporalOrdersDict:
                    temporalOrdersDict[key_] = [0, 0, 0]
                if int(record1[attrsMap['timestamp']]) < int(record2[attrsMap['timestamp']]):
                    temporalOrdersDict[key_][0] += 1
                elif int(record1[attrsMap['timestamp']]) > int(record2[attrsMap['timestamp']]):
                    temporalOrdersDict[key_][1] += 1
                else:
                    temporalOrdersDict[key_][2] += 1
                ''' 
                key_ = timelinessAttr + LIMITOR + k2 + LIMITOR + k1
                if key_ not in temporalOrdersDict:
                    temporalOrdersDict[key_] = [0, 0]
                if record2[attrsMap['timestamp']] < record1[attrsMap['timestamp']]:
                    temporalOrdersDict[key_][0] += 1
                else:
                    temporalOrdersDict[key_][1] += 1
                '''
    return temporalOrdersDict


def discoverCCsInTrain(dataClusters, schemas, attrsMap, data_np, timelinessAttrs, trainTOs):
    temporalOrdersDict = defaultdict(list)
    for index, record in enumerate(trainTOs):
        attr, eid = record[0], record[-1]
        val_1, val_2 = record[1], record[2] # val_1 > val_2
        key_1 = attr + LIMITOR + str(val_1[1]) + LIMITOR + str(val_2[1]) # '>'
        if key_1 not in temporalOrdersDict:
            # temporalOrdersDict[key_1] = [0, 0]
            temporalOrdersDict[key_1] = [[], []]
        # temporalOrdersDict[key_1][1] += 1
        temporalOrdersDict[key_1][1].append([val_1[0], val_2[0]])
        key_2 = attr + LIMITOR + str(val_2[1]) + LIMITOR + str(val_1[1]) # '<'
        if key_2 not in temporalOrdersDict:
            # temporalOrdersDict[key_2] = [0, 0]
            temporalOrdersDict[key_2] = [[], []]
        # temporalOrdersDict[key_2][0] += 1
        temporalOrdersDict[key_2][0].append([val_1[0], val_2[0]])
    return temporalOrdersDict


data_type = sys.argv[1]
timelinessAttrs = sys.argv[2]
CCs_file = sys.argv[3]
supp = int(sys.argv[4])


timelinessAttrs = [e.strip() for e in timelinessAttrs.split(',')]

data_file = os.path.join('../../data', data_type, 'data.csv')

# load training data
with open(os.path.join('../../data', data_type, 'training_processed.pkl'), 'rb') as f:
    trainTOs = pickle.load(f)
np.random.seed(30)
sample_ratio = 0.5
sc = np.random.choice(len(trainTOs), int(sample_ratio * len(trainTOs)), replace=False)
_list_processed = []
for e in sc:
    ll = trainTOs[e]
    if ll[1][0] <= ll[2][0]:
        continue
    _list_processed.append(trainTOs[e])
trainTOs = _list_processed

data = pd.read_csv(data_file)

dataClusters, schemas, attrsMap, data_np = clusterData(data)

# temporalOrdersDict = discoverCCs(dataClusters, schemas, attrsMap, data_np, timelinessAttrs)
temporalOrdersDict = discoverCCsInTrain(dataClusters, schemas, attrsMap, data_np, timelinessAttrs, trainTOs)

# to: [attr, val1, val2]
def transformCCs(to, id_1, id_2):
    attr, val_1, val_2 = to
    return 'data.t0.id == ' + str(id_1) + ' ^ data.t1.id == ' + str(id_2) + ' ^ ' + 'data.t0.' + attr + ' == ' + val_1 + ' ^ data.t1.' + attr + ' == ' + val_2 + ' -> ' 'data.t0.' + attr + ' < ' + 'data.t1.' + attr     

confThreshold = 1.0
CCs = []
for to, stat in temporalOrdersDict.items():
    if len(stat[-1]) > 0:
        continue
    conf = len(stat[0]) * 1.0 / ((len(stat[0]) + len(stat[1])))
    if conf >= confThreshold and len(stat[0]) >= supp:
        to_ = to.split(LIMITOR)
        for p in stat[0]:
            CCs.append(transformCCs(to_, p[0], p[1]))


'''
# to: [attr, val1, val2]
def transformCCs(to):
    attr, val_1, val_2 = to
    return 'data.t0.' + attr + ' == ' + val_1 + ' ^ data.t1.' + attr + ' == ' + val_2 + ' -> ' 'data.t0.' + attr + ' < ' + 'data.t1.' + attr     


confThreshold = 1.0
CCs = []
for to, stat in temporalOrdersDict.items():
    if stat[-1] > 0:
        continue
    conf = stat[0] * 1.0 / (sum(stat[:2]))
    if conf >= confThreshold and sum(stat) >= supp:
        to_ = to.split(LIMITOR)
        CCs.append(transformCCs(to_))
'''

f = open(CCs_file, 'w')
for cc in CCs:
    f.write(cc)
    f.write('\n')
f.close()
        





