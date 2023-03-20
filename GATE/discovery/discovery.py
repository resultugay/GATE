import numpy as np
import pandas as pd
import itertools
import argparse
import pickle
import copy
from collections import defaultdict
from sklearn.tree import _tree
from sklearn import preprocessing, tree

import sys
sys.path.append('../')
from critic.Fixes import Fixes, TemporalOrder, Operator
from critic.Utility import Status
from critic.Valuation import Predicate
from gate import Gate

def featureGeneration(predicatesArr, dataGroup, attrsMap, fixes, usedEIDs):
    features = []
    for eid, group in dataGroup.items():
        if eid not in usedEIDs:
            continue
        record_num = len(group)
        comb = itertools.permutations(np.arange(record_num), 2)
        for pair in comb:
            r_0, r_1 = group[pair[0]], group[pair[1]]
            tid_0, tid_1 = r_0[0], r_1[0]
            time_0, time_1 = r_0[-1], r_1[-1]
            if tid_0 == tid_1 or time_0 == time_1:
                continue
            fea = np.zeros(len(predicatesArr))
            for pid, predicate in enumerate(predicatesArr):
                if type(predicate) == 'str':
                    attr = predicate.split('.')[-1].strip()
                    aid = attrsMap[attr]
                    if predicate.find('t0') != -1:
                        fea[pid] = r_0[aid]
                    elif predicate.find('t1') != -1:
                        fea[pid] = r_1[aid]
                    continue
                if predicate.constant != None and predicate.constant.strip() == 'wildCard':
                    attr = predicate.operand1['attribute']
                    aid = attrsMap[attr]
                    if predicate.index1 == 't0':
                        fea[pid] = r_0[aid]
                    elif predicate.index2 == 't1':
                        fea[pid] = r_1[aid]
                if predicate.ifSatisfy(tid_0, tid_1, r_0, r_1, eid, attrsMap, fixes) == Status.SUCCESS:
                    fea[pid] = 1.0
            features.append(fea)
    return features

def toString(relation_name_1, attr_name_1, relation_name_2, attr_name_2, op, constant):
    #return relation_name + "->" + attr_name + ";" + str(constant) + ";" + "==";
    if constant == None:
        return relation_name_1 + '.' + 't0' + '.' + attr_name_1 + ' ' + op + ' ' + relation_name_2 + '.' + 't1' + '.' + attr_name_2
    else:
        return relation_name_1 + '.' + 't0' + '.' + attr_name_1 + ' ' + op + ' ' + constant


def genConstantPredicates(col, min_support_ratio, max_support_ratio, relation_name, attr_name):
    mapCount = {}
    for e in col:
        if e in mapCount:
            mapCount[e] += 1
        else:
            mapCount[e] = 1
    # calculate supports
    taumin = min_support_ratio * len(col)
    taumax = max_support_ratio * len(col)

    res = []
    for k, v in mapCount.items():
        if v >= taumin and v <= taumax:
            res.append(k)

    return [toString(relation_name, attr_name, relation_name, attr_name, '==', str(e)) for e in res]


''' data is pandas format
'''
def predicatesGeneration(data, relation_name, min_freq, max_freq, timelinessAttrs):
    # constant predicates
    allPredicates = []
    columns = list(data.columns)
    for col in columns:
        if col == 'timestamp' or col == 'id' or col == 'row_id':
            continue
        if data[col].dtype == 'O' or data[col].dtype == 'S':
            allPredicates += genConstantPredicates(data[col].values, min_freq, max_freq, relation_name, col)
        else:
            allPredicates += [relation_name + '.t0.' + col + ' == wildCard'  , relation_name + '.t1.' + col + ' == wildCard']
    # non-constant predicates
    for col in columns:
        if col == 'timestamp' or col == 'id' or col == 'row_id':
            continue
        # equal
        eq = toString(relation_name, col, relation_name, col, '==', None)
        allPredicates.append(eq)
        if data[col].dtype == 'int' or data[col].dtype == 'float' or data[col].dtype == 'double' or col in timelinessAttrs:
            # less than
            leq = toString(relation_name, col, relation_name, col, '<', None)
            # greater than
            geq = toString(relation_name, col, relation_name, col, '>', None)
            allPredicates += [leq, geq]

    return allPredicates

def transformTOs(dataProcessed):
    '''
    Transform data_processed to a list of TemporalOrder
    :param dataProcessed: a list of [attribute, (tid_0, value_0), (tid_1, value_1)]
    :return:
    '''
    tos = []
    for attr, pos, neg, eid in dataProcessed:
        to = TemporalOrder(pos[0], neg[0], attr, Operator.GREATOR_THAN, eid, 1.0)
        tos.append(to)
    return tos

def timelinessPredicateIDs(allPredicates, timelinessAttrs):
    resPIDs = []
    for pid, predicate in enumerate(allPredicates):
        if predicate.isTimelinessPredicate() and predicate.operator == '<':
            resPIDs.append(pid)
    return resPIDs


def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)


def tree_to_code_new(tree, feature_names, rhsPredicate, confThreshold):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    path, paths = [], []
    def recurse(node, depth, path, paths):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [name + ' <= ' + str(threshold)]
            #print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1, p1, paths)
            p2 += [name + ' > ' + str(threshold)]
            #print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1, p2, paths)
        else:
            #print("{}return {}".format(indent, tree_.value[node]))
            conf = tree_.value[node][0][0] * 1.0 / (tree_.value[node][0][0] + tree_.value[node][0][1])
            if conf < confThreshold:
                return
            ss = '  ^ '.join(path)
            ss += ' -> ' + rhsPredicate.toString()
            paths += [ss]

    recurse(0, 1, path, paths)
    return paths


'''
def mining(features, allPredicates, rhsPredicates, supportThreshold, confThreshold):
    def verify(features, Psel, rhsPredicates):
        

    def recursive(features, remainingPredicates, rhsPredicates, Psel, supportThreshold, confThreshold):
'''

def discoverSpeicalCCs(dataGroup, timelinessAttrs, attrsMap, relation, suppThreshold, confThreshold):
    def genOrderCmp(v_2, v_1):
        return str(v_2) + '<' + str(v_1)
    def genOrder(v_1, v_2):
        v_1, v_2 = min(v_1, v_2), max(v_1, v_2)
        return str(v_1) + '---' + str(v_2)

    ordersCmp, orders = defaultdict(int), defaultdict(int)
    for eid, records in dataGroup.items():
        records_ = sorted(records, key = lambda x : x[-1], reverse=True)
        for col in timelinessAttrs:
            aid = attrsMap[col]
            for pair in itertools.combinations(np.arange(len(records_)), 2):
                rid0, rid1 = min(pair[0], pair[1]), max(pair[0], pair[1])
                key = genOrderCmp(records_[rid1][aid], records_[rid0][aid])
                ordersCmp[key] += 1
                key_ = genOrder(records_[rid0][aid], records_[rid1][aid])
                orders[key_] += 1
    resCCs = []
    for k, v in ordersCmp.items():
        v_1, v_2 = k.split('<')[0].strip(), k.split('<')[1].strip()
        count = orders[genOrder(v_1, v_2)]
        if v * 1.0 / (count + 1e-9) >= confThreshold and v >= suppThreshold:
            print(k)



def main():
    parser = argparse.ArgumentParser(description='CCs discovery')
    parser.add_argument('--data_dir', type=str, help='data path')
    parser.add_argument('--min_freq', type=float, help='minimal support')
    parser.add_argument('--max_freq', type=float, help='maximal support')
    parser.add_argument('--relation', type=str, help='relational name')
    parser.add_argument('--timelinessAttrs', type=str, help='timeliness attributes')
    parser.add_argument('--dataProcessedFile', type=str, help='data processed (temporal orders) file')

    args = parser.parse_args()
    arg_dict = args.__dict__

    data = pd.read_csv(arg_dict['data_dir'])
    # load data processed file
    with open(arg_dict['dataProcessedFile'], 'rb') as f:
        dataProcessed = pickle.load(f)

    usedEIDs = []
    for record in dataProcessed:
        usedEIDs.append(record[-1])
    usedEIDs = set(usedEIDs)

    timelinessAttrs = [e.strip() for e in arg_dict['timelinessAttrs'].split(',')]

    tos = transformTOs(dataProcessed)
    fixes = Fixes()
    for to in tos:
        fixes.insert(to)

    allPredicatesStr = predicatesGeneration(data, arg_dict['relation'], arg_dict['min_freq'], arg_dict['max_freq'], timelinessAttrs)
    allPredicates = []
    for predicate_str in allPredicatesStr:
        p = Predicate(predicate_str, timelinessAttrs)
        allPredicates.append(p)

    dataGroup = defaultdict(list)
    for tid, record in enumerate(data.values):
        eid = record[1]
        dataGroup[eid].append(record)
    attrsMap = defaultdict(int)
    for aid, attr in enumerate(list(data.columns)):
        attrsMap[attr] = aid
    # generate features
    features = featureGeneration(allPredicates, dataGroup, attrsMap, fixes, usedEIDs)

    # extract temporal orders
    suppThreshold = 10
    confThreshold = 0.8
    discoverSpeicalCCs(dataGroup, timelinessAttrs, attrsMap, arg_dict['relation'], suppThreshold, confThreshold)

    # discover rules
    max_depth = 7
    clf = tree.DecisionTreeClassifier(max_depth=max_depth)

    schemas = []
    for predicate in allPredicates:
        if type(predicate) == 'str':
            schemas += [" ( " + predicate + " ) "]
        else:
            schemas += [" ( " + predicate.toString() + " ) "] #[" ( " + predicate.toString() + " ) " for predicate in allPredicates]
    features = np.array(features)

    # RHS application
    rhsPIDs = timelinessPredicateIDs(allPredicates, timelinessAttrs)

    CCs = []
    for rhspid in rhsPIDs:
        print('=======================================================================================================================')
        print('RHS predicate is {} '.format(allPredicates[rhspid].toString()))
        #train_X = np.concatenate((features[:, :rhspid], features[:, rhspid + 1:]), 1)
        train_X = copy.deepcopy(features)
        train_X[:, rhspid] = 0
        train_Y = features[:, rhspid][:, np.newaxis]
        clf = clf.fit(train_X, train_Y)
        CCs += tree_to_code_new(clf, schemas, allPredicates[rhspid], 0.7)

    print()
    print()
    print()

    # all CCs
    for cc in CCs:
        print(str(cc))


if __name__ == "__main__":
    main()

