from collections import defaultdict
from .Fixes import TemporalOrder, Operator


class PList(object):
    def __init__(self):
        self._list = []
        self.start = 0
    def add(self, vid):
        self._list.append(vid)
    def remove(self, vid, scid):
        self._list[scid] = self._list[self.start]
        self.start += 1

class GlobalStructures(object):
    '''
    def __init__(self, valuations_part, valuations_comp, indexValuations_part, indexValuations_comp, fixes, deltaTOs):
        self.valuations_part = valuations_part
        self.valuations_comp = valuations_comp
        self.indexValuations_part = indexValuations_part
        self.indexValuations_comp = indexValuations_comp
        self.fixes = fixes
        self.deltaTOs = deltaTOs
    '''

    def __init__(self, H, I, M, RHS, BT, TOs):
        '''
        :param H: the list of all valuations (complete valuation will be marked as NULL)
        :param I: the inverted index of H
        :param M: Hash of all calculated valuations
        :param RHS: the list of HANDLED temporal orders
        '''
        self.H = H
        self.I = I
        self.M = M
        self.RHS = RHS

        # used to recursively find all previous valuations, storing VIDs
        self.BT = BT
        self.stableTOs = TOs
        self.stableTOsIndex = defaultdict(list)

    def __init__(self):
        self.H = []
        self.I = defaultdict(PList)
        self.M = defaultdict(int)
        self.RHS = defaultdict()
        self.BT = defaultdict(list)
        self.stableTOs = defaultdict()
        self.stableTOsIndex = defaultdict(list)

    def isComputedV(self, k_v):
        if k_v in self.M:
            return True
        return False

    def update(self, newH):
        startVID = len(self.H)
        for h in newH:
            # update to H
            self.H.append(h)
            # update to I
            for to_key in h.getTOKeys():
                self.I[to_key].add(len(self.H) - 1)
            # update M
            kv = h.encode()
            self.M[kv] += 1

    def remove(self, vid, scid):
        h = self.H[vid]
        for to_key in h.getTOKeys():
            self.I[to_key].remove(vid, scid)
        self.H[vid] = None

    def updateBT(self, vid, listOfVIDs):
        self.BT[vid] += listOfVIDs

    def retrieveAllRelatedTOs(self, to):
        # 1. find all valuations whose RHS is to
        Hs_vids = []
        for vid, h in enumerate(self.H):
            if h == None:
                continue
            if h.getRHSTo().equal(to):
                Hs_vids.append(vid)
        # 2. retrieve all TOs
        res = []
        for vid in Hs_vids:
            self.findAllTOs(vid, self.BT, self.RHS, res)
        return res

    def findAllTOsBT(self, vid, BT, RHS, res):
        to = self.H[vid].getRHSTo()
        [t0, t1] = to.getPair()
        if str([t0, t1]) in RHS or str([t1, t0]) in RHS:
            return
        res.append(to)
        if vid not in BT:
            return
        for vid_ in BT[vid]:
            self.findAllTOs(vid_, BT, RHS, res)

    def findValuationsByTO(self, to):
        toKey = to.encode()
        plist = self.I[toKey]
        if plist == None:
            return 0, []
        return plist.start, plist._list

    def getValuation(self, vid):
        return self.H[vid]

    def generateKeyStableTosIndex(self, to):
        return str(to.eid) + '---' + str(to.attribute)

    def generateKeyStableTosIndex_(self, attr, eid):
        return str(eid) + '---' + str(attr)

    def addTO(self, to):
        self.stableTOs[to.encode()] = to
        # generate key
        key = self.generateKeyStableTosIndex(to)
        self.stableTOsIndex[key].append(to.encode())

    def addTOs(self, tos):
        for to in tos:
            self.stableTOs[to.encode()] = to
            key = self.generateKeyStableTosIndex(to)
            self.stableTOsIndex[key].append(to.encode())

    def generateTransitivity(self, to_new):
        res = []
        toKeys = self.stableTOsIndex[self.generateKeyStableTosIndex(to_new)]
        for toKey in toKeys:
            to = self.stableTOs[toKey]
            # 1. check to_new = (t0, t1) and to = (t1, t2) => to_trans = (t0, t2)
            if to_new.t1 == to.t0 and to_new.t0 != to.t1:
                to_trans = TemporalOrder(to_new.t0, to.t1, to.attribute, to.op, to.eid, 1.0)
                if self.isStableTO(to_trans):
                    continue
                res.append(to_trans)
            # 2. check to = (t0, t1) and to_new = (t1, t2) => to_trans = (t0, t2)
            if to.t1 == to_new.t0 and to.t0 != to_new.t1:
                to_trans = TemporalOrder(to.t0, to_new.t1, to.attribute, to.op, to.eid, 1.0)
                if self.isStableTO(to_trans):
                    continue
                res.append(to_trans)
        return res

    def sortGlobalOrder(self, attr, eid):
        toKeys = self.stableTOsIndex[self.generateKeyStableTosIndex_(attr, eid)]
        tos = []
        for toKey in toKeys:
            tos.append(self.stableTOs[toKey])
        tids = defaultdict()
        for to in tos:
            if to.t0 not in tids:
                tids[to.t0] = 1
            if to.t1 not in tids:
                tids[to.t1] = 1
        flag = True
        while flag:
            flag = False
            for to in tos:
                if tids[to.t1] <= tids[to.t0]:
                    tids[to.t1] = tids[to.t0] + 1
                    flag = True
            if not flag:
                break
        tids_arr = [[k, v] for k, v in tids.items()]
        tids_arr = sorted(tids_arr, key=lambda x : x[1], reverse=True)
        return [e[0] for e in tids_arr]

    def isValidTO(self, to):
        if to.encodeInverse() in self.stableTOs:
            return False
        return True

    # RHS, check whether TO has been placed into the stable set
    def isStableTO(self, to):
        if to.encodeInverse() in self.stableTOs or to.encode() in self.stableTOs:
            return True
        return False


    # 2 -- True, 1 -- not sure, 0 -- False
    def predictLabel(self, t_0, t_1, attr, eid):
        to = TemporalOrder(t_0, t_1, attr, Operator.LESS_THAN, eid, 0.0)
        if to.encode() in self.stableTOs: #and self.stableTOs[to.encode()].conf == 1.0:
            return 2
        if to.encodeInverse() in self.stableTOs: #and self.stableTOs[to.encodeInverse()].conf == 1.0:
            return 0
        return 1
