from sklearn.metrics import ndcg_score
import torch
from torchmetrics.functional import retrieval_reciprocal_rank, retrieval_normalized_dcg
from collections import defaultdict
import itertools
import numpy as np

''' ground_truth = [3, 5, 7, 10, 2] => [1, 2, 3, 4, 5]
    prediction = [3, 5, 10, 7, 2] => [1, 2, 4, 3, 5]
'''
def metrics(prediction_order, ground_truth_order, attribute, ifInTrainMap, _seed=20):
    '''
    m = defaultdict(int)
    ground_truth = np.arange(len(ground_truth_order)) + 1
    prediction = []
    for i, tid in enumerate(ground_truth_order):
         m[tid] = ground_truth[i]
    pred = [m[tid] for tid in prediction_order]
    '''
    tidsMap = defaultdict()
    tidsMapReverse = defaultdict()
    for i, e in enumerate(ground_truth_order):
        tidsMap[e] = i
        tidsMapReverse[i] = e
    ground_truth = [0] * len(ground_truth_order)
    for i, tid in enumerate(ground_truth_order):
        ground_truth[tidsMap[tid]] = np.exp(len(ground_truth_order) - i - 1) - 1
    pred = [0] * len(ground_truth_order)
    for i, tid in enumerate(prediction_order):
        pred[tidsMap[tid]] = np.exp(len(prediction_order) - i - 1) - 1

    # NDCG
    ndcgmetric = retrieval_normalized_dcg(torch.tensor(pred, dtype=torch.float32),
                                          torch.tensor(ground_truth))
    # MRR
    mrr_pred = [0] * len(pred)
    #mrr_pred[np.argmin(pred)] = 1
    mrr_pred[np.argmax(pred)] = 1
    mrr_ground_truth = [True] + [False] * (len(pred) - 1)
    mrr_pred = torch.tensor(mrr_pred, dtype=torch.float32)
    mrr_ground_truth = torch.tensor(mrr_ground_truth)
    mrr = retrieval_reciprocal_rank(mrr_pred, mrr_ground_truth)

    # TP, FP, FN
    np.random.seed(_seed)
    np.random.shuffle(pred)
    np.random.seed(_seed)
    np.random.shuffle(ground_truth)
    TP, FP, FN, acc, count = 0, 0, 0, 0, 0

    ground_truth_new = [[tidsMapReverse[i], e] for i, e in enumerate(ground_truth)]
    np.random.seed(_seed)
    np.random.shuffle(ground_truth_new)

    # only compare the latest one with others
    latest_sc = np.argmax(ground_truth)
    for i in range(len(ground_truth)):
        if i == latest_sc:
            continue

        sc0, sc1 = i, latest_sc

        ttid_0, ttid_1 = ground_truth_new[sc0][0], ground_truth_new[sc1][0]
        if isTrainingKey(ttid_0, ttid_1, attribute) in ifInTrainMap:
            continue


        if i > latest_sc:
            z_ = sc0
            sc0 = sc1
            sc1 = z_
        label, prediction = ground_truth[sc0] < ground_truth[sc1], pred[sc0] < pred[sc1]
        # TP
        if label == True and prediction == True:
            TP += 1
        # FP
        if prediction == True and label == False:
            FP += 1
        # FN
        if label == True and prediction == False:
            FN += 1
        # acc
        if label == prediction:
            acc += 1
        count += 1

    '''
    for pair in itertools.combinations(np.arange(len(pred)), 2):
        sc0, sc1 = pair[0], pair[1]
        if sc0 == sc1:
            continue
        label, prediction = ground_truth[sc0] < ground_truth[sc1], pred[sc0] < pred[sc1]
        # TP
        if label == True and prediction == True:
            TP += 1
        # FP
        if prediction == True and label == False:
            FP += 1
        # FN
        if label == True and prediction == False:
            FN += 1
        # acc
        if label == prediction:
            acc += 1
        count += 1
    '''
    return ndcgmetric, mrr, TP, FP, FN, acc, count

def metrics_(prediction_order, ground_truth_order, attribute, ifInTrainMap):
    '''
    m = defaultdict(int)
    ground_truth = np.arange(len(ground_truth_order)) + 1
    prediction = []
    for i, tid in enumerate(ground_truth_order):
         m[tid] = ground_truth[i]
    pred = [m[tid] for tid in prediction_order]
    '''
    tidsMap = defaultdict()
    tidsMapReverse = defaultdict()
    for i, e in enumerate(ground_truth_order):
        tidsMap[e] = i
        tidsMapReverse[i] = e
    ground_truth = [0] * len(ground_truth_order)
    for i, tid in enumerate(ground_truth_order):
        ground_truth[tidsMap[tid]] = np.exp(len(ground_truth_order) - i - 1) - 1
    pred = [0] * len(ground_truth_order)
    for i, tid in enumerate(prediction_order):
        pred[tidsMap[tid]] = np.exp(len(prediction_order) - i - 1) - 1

    # NDCG
    ndcgmetric = retrieval_normalized_dcg(torch.tensor(pred, dtype=torch.float32),
                                          torch.tensor(ground_truth))
    # MRR
    mrr_pred = [0] * len(pred)
    #mrr_pred[np.argmin(pred)] = 1
    mrr_pred[np.argmax(pred)] = 1
    mrr_ground_truth = [True] + [False] * (len(pred) - 1)
    mrr_pred = torch.tensor(mrr_pred, dtype=torch.float32)
    mrr_ground_truth = torch.tensor(mrr_ground_truth)
    mrr = retrieval_reciprocal_rank(mrr_pred, mrr_ground_truth)

    # TP, FP, FN
    np.random.seed(20)
    np.random.shuffle(pred)
    np.random.seed(20)
    np.random.shuffle(ground_truth)
    TP, FP, FN, acc, count = 0, 0, 0, 0, 0

    ground_truth_new = [[tidsMapReverse[i], e] for i, e in enumerate(ground_truth)]
    np.random.seed(20)
    np.random.shuffle(ground_truth_new)

    # only compare the latest one with others
    latest_sc = np.argmax(ground_truth)
    for i in range(len(ground_truth)):
        if i == latest_sc:
            continue

        sc0, sc1 = i, latest_sc

        ttid_0, ttid_1 = ground_truth_new[sc0][0], ground_truth_new[sc1][0]
        if isTrainingKey(ttid_0, ttid_1, attribute) in ifInTrainMap:
            continue


        if i > latest_sc:
            z_ = sc0
            sc0 = sc1
            sc1 = z_
        label, prediction = ground_truth[sc0] < ground_truth[sc1], pred[sc0] < pred[sc1]
        # TP
        if label == True and prediction == True:
            TP += 1
        # FP
        if prediction == True and label == False:
            FP += 1
        # FN
        if label == True and prediction == False:
            FN += 1
        # acc
        if label == prediction:
            acc += 1
        count += 1

    '''
    for pair in itertools.combinations(np.arange(len(pred)), 2):
        sc0, sc1 = pair[0], pair[1]
        if sc0 == sc1:
            continue
        label, prediction = ground_truth[sc0] < ground_truth[sc1], pred[sc0] < pred[sc1]
        # TP
        if label == True and prediction == True:
            TP += 1
        # FP
        if prediction == True and label == False:
            FP += 1
        # FN
        if label == True and prediction == False:
            FN += 1
        # acc
        if label == prediction:
            acc += 1
        count += 1
    '''
    return ndcgmetric, mrr, TP, FP, FN, acc, count

def isTrainingKey(t0, t1, attr):
    return str(t0) + '---' + str(t1) + '---' + str(attr)

def FScore(dists, ground_truth_order, attribute, eid, GS, ifInTrainMap):

    '''
    tidsMap = defaultdict()
    for i, e in enumerate(ground_truth_order):
        tidsMap[e] = i
    ground_truth = [0] * len(ground_truth_order)
    for i, tid in enumerate(ground_truth_order):
        ground_truth[tidsMap[tid]] = np.exp(len(ground_truth_order) - i - 1) - 1
    pred = [0] * len(ground_truth_order)
    for i, tid in enumerate(ground_truth_order):
        pred[tidsMap[tid]] = -dists[i] #np.exp(len(prediction_order) - i - 1) - 1

    TP, FP, FN, acc, count = 0, 0, 0, 0, 0

    # only compare the latest one with others
    latest_sc = np.argmax(ground_truth)
    for i in range(len(ground_truth)):
        if i == latest_sc:
            continue
        sc0, sc1 = i, latest_sc
        if i > latest_sc:
            z_ = sc0
            sc0 = sc1
            sc1 = z_

        t0, t1 = ground_truth_order[sc0], ground_truth_order[sc1]
        if isTrainingKey(t0, t1, attribute) in ifInTrainMap:
            continue
        label, prediction = ground_truth[sc0] < ground_truth[sc1], pred[sc0] < pred[sc1]
        status = GS.predictLabel(t0, t1, attribute, eid)
        if status == 2:
            prediction = True
        elif status == 0:
            prediction = False
        # TP
        if label == True and prediction == True:
            TP += 1
        # FP
        if prediction == True and label == False:
            FP += 1
        # FN
        if label == True and prediction == False:
            FN += 1
        # acc
        if label == prediction:
            acc += 1
        count += 1
    '''

    TP, FP, FN, acc, count = 0, 0, 0, 0, 0
    tidsMap = defaultdict(list)
    for i, tid in enumerate(ground_truth_order):
        # ground truth score
        tidsMap[tid].append(len(ground_truth_order) - i)
        # prediction score
        tidsMap[tid].append(-dists[i])
    np.random.seed(20)
    np.random.shuffle(ground_truth_order)

    for pair in itertools.combinations(np.arange(len(ground_truth_order)), 2):
        sc0, sc1 = pair[0], pair[1]
        if sc0 == sc1:
            continue
        t0, t1 = ground_truth_order[sc0], ground_truth_order[sc1]
        if isTrainingKey(t0, t1, attribute) in ifInTrainMap:
            continue
        score0, pred0, score1, pred1 = tidsMap[t0][0], tidsMap[t0][1], tidsMap[t1][0], tidsMap[t1][1]
        label, prediction = score0 < score1, pred0 < pred1
        status = GS.predictLabel(t0, t1, attribute, eid)
        if status == 2:
            prediction = True
        elif status == 0:
            prediction = False
        # TP
        if label == True and prediction == True:
            TP += 1
        # FP
        if prediction == True and label == False:
            FP += 1
        # FN
        if label == True and prediction == False:
            FN += 1
        # acc
        if label == prediction:
            acc += 1
        count += 1


    return TP, FP, FN, acc, count

def FScoreCritic_(ground_truth_order, attribute, eid, GS, ifInTrainMap):
    TP, FP, FN, acc, count = 0, 0, 0, 0, 0
    tidsMap = defaultdict(list)
    for i, tid in enumerate(ground_truth_order):
        # ground truth score
        tidsMap[tid].append(len(ground_truth_order) - i)
        # prediction score
        tidsMap[tid].append(0)

    np.random.seed(20)
    np.random.shuffle(ground_truth_order)

    for pair in itertools.combinations(np.arange(len(ground_truth_order)), 2):
        sc0, sc1 = pair[0], pair[1]
        if sc0 == sc1:
            continue
        t0, t1 = ground_truth_order[sc0], ground_truth_order[sc1]
        if isTrainingKey(t0, t1, attribute) in ifInTrainMap:
            continue
        score0, score1 = tidsMap[t0][0], tidsMap[t1][0]
        label = score0 < score1
        #print('label : ', label)
        status = GS.predictLabel(t0, t1, attribute, eid)
        if status == 2:
            prediction = True
        else:
            prediction = False
        # TP
        if label == True and prediction == True:
            TP += 1
        # FP
        if prediction == True and label == False:
            FP += 1
        # FN
        if label == True and prediction == False:
            FN += 1
        # acc
        if label == prediction:
            acc += 1
        count += 1

    return TP, FP, FN, acc, count


def FScoreCritic(ground_truth_order, attribute, eid, GS, ifInTrainMap):
    TP, FP, FN, acc, count = 0, 0, 0, 0, 0
    tidsMap = defaultdict(list)
    for i, tid in enumerate(ground_truth_order):
        # ground truth score
        tidsMap[tid].append(len(ground_truth_order) - i)
        # prediction score
        tidsMap[tid].append(0)

    tid_latest = ground_truth_order[0]
    np.random.seed(20)
    np.random.shuffle(ground_truth_order)
    latest_sc = ground_truth_order.index(tid_latest)

    #for pair in itertools.combinations(np.arange(len(ground_truth_order)), 2):
    for i in range(len(ground_truth_order)):
        if i == latest_sc:
            continue
        sc0, sc1 = i, latest_sc
        if i > latest_sc:
            z_ = sc0
            sc0 = sc1
            sc1 = z_
        t0, t1 = ground_truth_order[sc0], ground_truth_order[sc1]
        if isTrainingKey(t0, t1, attribute) in ifInTrainMap:
            continue
        score0, score1 = tidsMap[t0][0], tidsMap[t1][0]
        label = score0 < score1
        # print('label : ', label)
        status = GS.predictLabel(t0, t1, attribute, eid)
        if status == 2:
            prediction = True
        elif status == 0:
            prediction = False
        else:
            count += 1
            continue
        # TP
        if label == True and prediction == True:
            TP += 1
        # FP
        if prediction == True and label == False:
            FP += 1
        # FN
        if label == True and prediction == False:
            FN += 1
        # acc
        if label == prediction:
            acc += 1
        count += 1

    return TP, FP, FN, acc, count