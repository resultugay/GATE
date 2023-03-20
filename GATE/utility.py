import numpy as np
import pandas as pd
from collections import defaultdict


def procComm_(original_comm__file):
    data = pd.read_csv(original_comm__file)
    # replace 'nan' to 0
    data.fillna(0)
    data_ = data[['entity_id', 'scope', 'name', 'address', 'owner']]
    data_np = np.array(data_.values, 'str')
    # data_np = np.array(data_.values)
    # data_np = np.unique(data_np, axis=0)
    data_np = np.array(data_np, 'str')

    print("The number of data is {} after removing duplicates".format(len(data_np)))

    uniques_id = []
    hist = defaultdict()
    for tid, record in enumerate(data_np):
        if str(record) not in hist:
            uniques_id.append(tid)
            hist[str(record)] = 0

    data_group = defaultdict(list)
    for sc in uniques_id:
        record = data_np[sc]
        data_group[record[0]].append(record)
    
    data_final, tid, eid_ = [], 0, 0
    entity_ids = sorted(np.array(list(data_group.keys()), 'int'))
    print('entity_ids')
    # for k, v in data_group.items():
    for eid in entity_ids:
        k = str(eid)
        v = data_group[k]
        if len(v) <= 1:
            continue
        t = 0
        for r in v:
            # data_final.append([tid] + [r[0]] + [tid] + list(r[1:]) + [t])
            data_final.append([tid] + [eid_] + [tid] + list(r[1:]) + [t])
            tid += 1
            t += 1
        eid_ += 1
    data_pd = pd.DataFrame(data_final, columns=list(data.columns))
    data_pd = data_pd.fillna(0)
    return data_pd




