import logging
import itertools
import numpy as np
from .GS import GlobalStructures
from .Chase import Chase
import sys
sys.path.append('..')
from metrics import metrics, FScoreCritic

class Critic:
    '''
    def __init__(self):
        pass
    '''


    def __init__(self, D, schemas, timelinessAttrs, attrsMap):
        self.chase = Chase(D, schemas, timelinessAttrs)
        self.attrsMap = attrsMap

    def deduce(self, CCs, deltaTOStable, deltaTOML, indicator_last, GS, option, round=-1):
        sigma_, GS_, indicator_ = None, None, None
        if option == 'gatenc':
            sigma_, GS_, indicator_ = self.chase.ChaseNC(CCs, GS, deltaTOStable, deltaTOML,
                                                      indicator_last, self.chase.D, self.attrsMap)
        else:
            #sigma_, GS_, indicator_ = self.chase.Chase(CCs, GS, deltaTOStable, deltaTOML,
                                                    # indicator_last, self.chase.D, self.attrsMap)
            sigma_, GS_, indicator_ = self.chase.ChaseAndRemoveTOML(CCs, GS, deltaTOStable, deltaTOML,
                                                     indicator_last, self.chase.D, self.attrsMap, round)
        return sigma_, GS_, indicator_

    def generateFullOrder(self, data, GS):
        attr, _list, eid = data[0], data[1], data[2]
        ground_truth = [e[0] for e in _list]
        tids_detect = GS.sortGlobalOrder(attr, eid)
        np.random.seed(100)
        sc = np.random.choice(len(_list), len(tids_detect), replace=False)
        sc = sorted(sc)
        pred = [-1] * len(_list)
        for i, z in enumerate(sc):
            pred[z] = tids_detect[i]
        tids_others = [e[0] for e in _list if e[0] not in tids_detect]
        np.random.seed(50)
        np.random.shuffle(tids_others)
        ss = 0
        for i in range(len(pred)):
            if pred[i] == -1:
                pred[i] = tids_others[ss]
                ss += 1
        '''
        sc_others = [i for i in range(len(_list)) if i not in sc]
        np.random.seed(30)
        np.random.shuffle(sc_others)
        ss = 0
        for i in range(len(pred)):
            if pred[i] == -1:
                pred[i] = _list[sc_others[ss]][0]
                ss += 1
        '''
        return ground_truth, pred

    def evaluationValid(self, validation_data, GS, round, ifInTrainMap):
        ndcgmetric, mrr, TP, FP, FN, acc, count = 0, 0, 0, 0, 0, 0, 0
        TP_, FP_, FN_, acc_, count_ = 0, 0, 0, 0, 0
        validation_processed = validation_data['data_processed']
        for index, data in enumerate(validation_processed):
            ground_truth, pred = self.generateFullOrder(data, GS)
            # compared ground truth, and pred
            ndcg_s, mrr_s, TP_s, FP_s, FN_s, acc_s, count_s = metrics(pred, ground_truth, data[0], ifInTrainMap, index * 100)
            ndcgmetric += ndcg_s
            mrr += mrr_s
            TP += TP_s
            FP += FP_s
            FN += FN_s
            acc += acc_s
            count += count_s

            TP_ss, FP_ss, FN_ss, acc_ss, count_ss = FScoreCritic(ground_truth, data[0], data[-1], GS, ifInTrainMap)
            TP_ += TP_ss
            FP_ += FP_ss
            FN_ += FN_ss
            acc_ += acc_ss
            count_ += count_ss

        ndcg = ndcgmetric * 1.0 / len(validation_processed)
        mrr = mrr * 1.0 / len(validation_processed)
        precision = TP * 1.0 / (TP + FP)
        recall = TP * 1.0 / (TP + FN)
        Fmeasure = 2 * precision * recall / (precision + recall)
        accuracy = acc * 1.0 / count #len(validation_processed)

        logging.info("round={} ndcg={} mrr={} precision={} recall={} Fmeasure={}".format(round, ndcg, mrr, precision, recall, Fmeasure))
        print("round={} ndcg={} mrr={} precision={} recall={} Fmeasure={} accuracy={}".format(round, ndcg, mrr, precision, recall, Fmeasure, accuracy))

        precision = TP_ * 1.0 / (TP_ + FP_ + 1e-10)
        recall = TP_ * 1.0 / (TP_ + FN_ + 1e-10)
        Fmeasure = 2 * precision * recall / (precision + recall + 1e-10)
        accuracy = acc_ * 1.0 / count_ #len(validation_processed)
        print("roundFScore={} precision={} recall={} Fmeasure={} accuracy={}".format(round, precision, recall, Fmeasure, accuracy))


    def deduce_(self, high_conf_css, training_data, ccs):

        if not high_conf_css:
            return None

        new_ccs = {}
        latest_val = ''
        non_latest_val = ' '
        new_latest = ''
        new_non_latest = ' '
        for attr1, attr2 in ccs:
            for eid in training_data.entity_id.unique():
                sub_df = training_data.loc[training_data['entity_id'] == eid]
                rows = sub_df[[str(attr1), str(attr2), 'timestamp']].to_dict('Records')
                comb = itertools.combinations(rows, 2)
                for t1, t2 in comb:
                    if t1[attr2] != t2[attr2] and t1['timestamp'] != t2['timestamp']:
                        if t1['timestamp'] > t2['timestamp']:
                            latest_val, non_latest_val = t1[attr1], t2[attr1]
                            new_latest, new_non_latest = t1[attr2], t2[attr2]
                        elif t1['timestamp'] < t2['timestamp']:
                            non_latest_val, latest_val = t1[attr1], t2[attr1]
                            new_non_latest, new_latest = t1[attr2], t2[attr2]
                        if attr1 in high_conf_css and (latest_val, non_latest_val) in high_conf_css[attr1]:
                            if attr2 not in new_ccs:
                                new_ccs[attr2] = {}
                            if (new_latest, new_non_latest) not in new_ccs[attr2]:
                                new_ccs[attr2][(new_latest, new_non_latest)] = 0
                            new_ccs[attr2][(new_latest, new_non_latest)] += 1
        for key in new_ccs.keys():
            new_ccs[key] = sorted(new_ccs[key].keys(), key=lambda x: x[1], reverse=True)

        return new_ccs

    def conflict(self, simple_ccs, high_conf_ccs):
        conflicts = {}
        for attribute, currency_constraints in high_conf_ccs.items():
            for latest,non_latest in currency_constraints:
                if attribute in simple_ccs:
                    if (str(non_latest), str(latest)) in simple_ccs[attribute]:
                        if attribute not in conflicts:
                            conflicts[attribute] = []
                        conflicts[attribute].append((non_latest, latest))
        return conflicts
