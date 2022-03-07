import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import itertools

df = pd.read_csv('data/test.csv')
df.index = df.id

import pickle

with open('cc.pickle', 'rb') as handle:
    cc = pickle.load(handle)

def load_vectors(col):
    try:
        vector = {}
        path = 'output_vectors/' + str(col) + '.txt'
        with open(path,'r') as f:
            for line in f:
                split_line = line.split('\t')
                word = split_line[0]
                embedding = np.array(split_line[1:-1], dtype=np.float64)
                vector[word] = embedding
        print(col," vector loaded")

        return vector
    except:
        print('could not load vector for ', col)
        return None

fn = 0
tp = 0
fp = 0
unknown = 0
columns = set(cc.keys())
for id_ in [1]:
    sub_df = df.loc[id_]
    sub_df.index = sub_df.row_id
    rows = sub_df.to_dict('Records')
    for i in itertools.combinations(rows,2):
        for col in cc.keys():
            for currency in cc[col]:
                reverse_pair = False
                not_candidate = False
                for c in currency[0]:
                    if c[0] not in ['salary','kids']:
                        value_tuple_1 = i[0][c[0]]
                        value_tuple_2 = i[1][c[0]]
                        if not reverse_pair:
                            if value_tuple_1 == c[1] and value_tuple_2 == c[2]:
                                pass
                            elif value_tuple_1 == c[2] and value_tuple_2 == c[1]:
                                reverse_pair = True
                            else:
                                not_candidate = True
                                break
                        else:
                            if value_tuple_1 == c[2] and value_tuple_2 == c[1]:
                                pass
                            else:
                                not_candidate = True
                                break
                    else:
                        value_tuple_1 = i[0][c[0]]
                        value_tuple_2 = i[1][c[0]]
                        if not reverse_pair:
                            if c[1] == '<':
                                if value_tuple_1 < value_tuple_2:
                                    pass
                                elif value_tuple_1 > value_tuple_2:
                                    reverse_pair = True
                                else:
                                    not_candidate = True
                                    break

                            elif c[1] == '=':
                                if value_tuple_1 == value_tuple_2:
                                    pass
                                else:
                                    not_candidate = True
                                    break
                            elif c[1] == '>':
                                if value_tuple_1 > value_tuple_2:
                                    pass
                                elif value_tuple_1 < value_tuple_2:
                                    reverse_pair = True
                                else:
                                    not_candidate = True
                                    break
                        else:
                            if c[1] == '<':
                                if value_tuple_1 > value_tuple_2:
                                    pass
                                elif value_tuple_1 < value_tuple_2:
                                    reverse_pair = True
                                else:
                                    not_candidate = True
                                    break
                            elif c[1] == '>':
                                if value_tuple_1 < value_tuple_2:
                                    pass
                                elif value_tuple_1 > value_tuple_2:
                                    reverse_pair = True
                                else:
                                    not_candidate = True
                                    break

                if not_candidate:
                    pass
                elif reverse_pair:
                    if i[0]['timestamp'] < i[1]['timestamp']:
                        tp += 1
                    else:
                        fp += 1
                    break
                else:
                    if i[0]['timestamp'] > i[1]['timestamp']:
                        tp += 1
                    else:
                        fp += 1
                    break

    if not_candidate:
        unknown += 1