from collections import defaultdict
from enum import Enum
from Utility import *

ETID_ATTR_DELIMITOR = '-'

class Operator(Enum):
    GREATOR_THAN = ">"
    LESS_THAN = "<"

class TemporalOrder(object):
    def __init__(self, t0, t1, attribute, op, eid, conf):
        self.t0 = t0
        self.t1 = t1
        self.attribute = attribute
        self.op = op
        self.eid = eid
        self.conf = conf

    def encode(self):
        return str(self.t0) + ETID_ATTR_DELIMITOR + str(self.t1) + ETID_ATTR_DELIMITOR + str(self.attribute) + '_' + str(self.eid)

    def getOp(self):
        return self.op

    def getConf(self):
        return self.conf

'''
    fixes store temporal orders
'''
class Fixes:
    def __init__(self):
        self.buckets_geq = defaultdict(list)     # key: EID-attribute, value: bucket
        self.buckets_leq = defaultdict(list)
        self.allTemporalOrders = defaultdict() # only store < temporal orders

    def encodeKey(self, attributeName, eid):
        return str(eid) + ETID_ATTR_DELIMITOR + attributeName

    def insert(self, eid, attribute, to):
        k = self.encodeKey(eid, attribute)
        ko = to.encode()
        op = to.getOp()
        if op == Operator.GREATOR_THAN:
            self.buckets_geq[k].append(ko)
        else:
            self.buckets_leq[k].append(ko)
        self.allTemporalOrders[ko] = to

    def detectConflict(self):
        L = defaultdict(set)
        for k in self.buckets_leq.keys():
            bucket_leq = buckets_leq[k]
            bucket_geq = buckets_geq[k]
            # check overlap
            L[k] += list(set(bucket_leq).intersection(set(bucket_geq)))
        return L

    def checkTemporalOrder(self, v_1, v_2, eid, attr):
        k = str(v_1) + ETID_ATTR_DELIMITOR + str(v_2) + ETID_ATTR_DELIMITOR + str(attr) + '_' + str(eid)
        k_ = str(v_2) + ETID_ATTR_DELIMITOR + str(v_1) + ETID_ATTR_DELIMITOR + str(attr) + '_' + str(eid)
        if k in self.allTemporalOrders:
            return Status.SUCCESS
        elif k_ in self.allTemporalOrders:
            return Status.FAILURE
        return Status.PENDING

    def checkTemporalOrder(self, v_1, v_2, eid, attr, selectedTOKeys):
        k = str(v_1) + ETID_ATTR_DELIMITOR + str(v_2) + ETID_ATTR_DELIMITOR + str(attr) + '_' + str(eid)
        k_ = str(v_2) + ETID_ATTR_DELIMITOR + str(v_1) + ETID_ATTR_DELIMITOR + str(attr) + '_' + str(eid)
        if k in self.allTemporalOrders:
            if selectedTOKeys != None:
                selectedTOKeys.append(k)
            return Status.SUCCESS
        elif k_ in self.allTemporalOrders:
            return Status.FAILURE
        return Status.PENDING

    def generateAndSaveTO(self, v_1, v_2, attr, op, eid, conf):
        if op == ">":
            t = v_1
            v_1 = v_2
            v_2 = t
        to = TemporalOrder(v_1, v_2, attr, "<", eid, conf)
        self.insert(eid, attr, to)
        return to

    def computeConfTOs(self, toKeys):
        conf = self.allTemporalOrders[toKeys[0]].getConf()
        for k in toKeys[1:]:
            conf = min(conf, self.allTemporalOrders[k].getConf())
        return conf




