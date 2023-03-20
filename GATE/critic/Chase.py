import copy
from collections import defaultdict
from .Valuation import *
from .GS import *

class Chase(object):

    ''' the first column is TID, the second column is EID
    '''
    def __init__(self, D, schemas, timelinessAttrs):
        self.D = D # original dataset (only consider ONE)
        self.schemas = schemas
        self.attrMap = defaultdict(int)
        for aid, sc in enumerate(self.schemas):
            self.attrMap[sc] = aid
        self.PLIs = []
        self.computedValuations = defaultdict()
        self.timelinessAttr = timelinessAttrs

    def constructPLIs(self):
        for cid in range(2, len(self.schemas), 1):
            pli = self.constructPLI(self.D, cid)
            self.PLIs.append(pli)

    def constructPLI(self, D, cid):
        pli = defaultdict(defaultdict(list))
        for tid in range(len(D)):
            eid = D[tid][1]
            c = D[tid][cid]
            pli[c][eid].append(tid)

    def envokeTOLHS(self, to, ccs, GS):
        valuations = []
        eid = to.getEID()
        attr = to.getAttr()
        v_1, v_2 = self.D[to.t0][self.attrMap[attr]], self.D[to.t1][self.attrMap[attr]]
        ccsIDList = ccs.getIndex()[attr]
        if ccsIDList == None:
            return valuations
        for cc_id in ccsIDList:
            CC = ccs.getCC(cc_id)
            t_0, t_1 = to.t0, to.t1
            v = Valuation(CC, cc_id, t_0, t_1, eid, [])
            key_v = v.encode()
            # check whether the valudation has been computed
            if GS.isComputedV(key_v):
                continue
            # check whether a TO is not correct
            if not v.checkTOPredicates_(GS):
                continue
            if not v.checkNonTOPredicates(self.attrMap, self.D):
                continue
            valuations.append(v)
        return valuations

    ''' only consider rules whose LHSs contain timeliness predicate
    '''
    def envokeTOLHS_(self, ccs, fixes):
        valuations = []
        for toKey in fixes.allTemporalOrders:
            to = fixes.allTemporalOrders[toKey]
            eid = to.getEID()
            v_1, v_2 = to.t_0, to.t_1
            attr = to.getAttr() # find the timeliness attribute of the temporal order
            t0List = self.PLIs[self.attrMap[attr]][v_1][eid]
            t1List = self.PLIs[self.attrMap[attr]][v_2][eid]
            # find CCs whose timeliness attributes contain "attr"
            ccsIDList = ccs.getIndex()[attr]
            if ccsIDList != None:
                for cc_id in ccsIDList:
                    CC = ccs.getCC(cc_id)
                    p_t = CC.getTPredicateInLHS(attr)
                    for t_0 in t0List:
                        for t_1 in t1List:
                            if t_0 >= t_1:
                                continue
                            v = Valuation(CC, cc_id, t_0, t_1, eid, [])
                            k_v = v.encode()
                            if k_v in self.computedValuations:
                                continue
                            if not v.checkTOPredicate(self.attrMap, self.D, fixes):
                                continue
                            if not v.checkNonTOPredicate(self.attrMap, self.D):
                                continue
                            self.computedValuations[k_v] = 1
                            valuations.append(v)
        return valuations

    ''' only consider regular CCs, s.t., timeliness predicate only appears in RHS
    '''
    def execute(self, ccs, fixes):
        TOs = []
        for cc_id in range(len(ccs.getCCsList)):
            cc = ccs.getCC(cc_id)
            p_repr = cc.getOneNonTOPredicate()   # get the representive predicate of LHS
            tps = p_repr.extractPredicateTuplePairsList(D, self.PLIs, self.attrMap)
            for eid, tpsList in tps.items():
                for tp in tpsList:
                    v = Valuation(cc, cc_id, tp[0], tp[1], eid, [])
                    if not v.checkNonTOPredicate(self.attrMap, self.D):
                        continue
                    # v is a complete one, deduce RHS
                    to = v.deduceRHS(self.attrMap, self.D, fixes)
                    TOs.append(to)
        return TOs

    ''' ccsList is a list of CCs
        CCs_0: ccs whose LHSs contain timeliness predicates
        CCs_1: ccs whose LHSs do not contain timeliness predicates
    '''
    def classifyCCs(self, ccsArr):
        CCs_0, CCs_1 = None, None
        CCsList_0, CCsList_1 = [], []
        for cc in ccsArr:
            if len(cc.getTimelinessPIDs()) == 0:
                CCsList_1.append(cc)
            else:
                CCsList_0.append(cc)
        CCs_0 = CCs(CCsList_0, self.timelinessAttr)
        CCs_1 = CCs(CCsList_1, self.timelinessAttr)
        return CCs_0, CCs_1

    ''' assume rules whose LHSs do not contain temporal orders are filtered
        deltaTOStable is the new stable list of temporal orders
        deltaTOML is the defaultdict, where the key is the encode and value is TO
    '''
    def Chase(self, CCs, GS, deltaTOStable, deltaTOML, indicator_last, D, attrsMap):
        fixes = Fixes()
        sigma = []
        if indicator_last == False:
            for to in deltaTOStable:
                sigma += GS.retrieveAllRelatedTOs(to)
        # add TOs to fixes
        for to in sigma:
            fixes.insert(to)
        delta = deltaTOStable + sigma
        while len(delta) > 0:
            delta_new = []
            for to in delta:
                newH = self.envokeTOLHS(to, CCs, GS)
                GS.update(newH)
                l_start, l_list = GS.findValuationsByTO(to)
                for i in range(l_start, len(l_list), 1):
                    vid = l_list[i]
                    # get the valuation
                    h = GS.getValuation(vid)
                    # check the RHS TO of h has been handled and placed in the stable set
                    if GS.isStableTO(h.getRHSTO()):
                        continue
                    # mark to as validate in h
                    h.addValidTO(to)
                    if h.isComplete():
                        # when complete, deduce RHS temporal order
                        to_rhs = h.deduceRHS(attrsMap, D, fixes)
                        GS.remove(vid, i)
                        # check conflict
                        if fixes.checkTemporalOrder_(to_rhs) == Status.FAILURE or to_rhs.encodeInverse() in deltaTOML:   # conflict
                            return [[to_rhs], GS, False]
                        # transitivity 1.0
                        '''
                        tos_trans_res = []
                        tos_trans = GS.generateTransitivity(to_rhs)
                        for to_trans in tos_trans:
                            if fixes.checkTemporalOrder_(to_trans) == Status.FAILURE or to_trans.encodeInverse() in deltaTOML: #conflict
                                return [[to_trans], GS, False]
                            tos_trans_res.append(to_trans)
                        tos_trans = fixes.generateTransitivity(to_rhs)
                        for to_trans in tos_trans:
                            if GS.isStableTO(to_trans):
                                continue
                            if fixes.checkTemporalOrder_(to_trans) == Status.FAILURE or to_trans.encodeInverse() in deltaTOML: # conflict
                                return [[to_trans], GS, False]
                            tos_trans_res.append(to_trans)

                        fixes.insert(to_rhs)
                        sigma.append(to_rhs)
                        delta_new.append(to_rhs)

                        # add transitivity temporal orders
                        for to_trans in tos_trans_res:
                            fixes.insert(to_trans)
                            sigma.append(to_trans)
                            delta_new.append(to_trans)
                        '''
                        # transitivity 2.0
                        res = self.transitivity(to_rhs, GS, fixes, deltaTOML, sigma, delta_new)
                        if res != None:
                            return res

            delta = copy.deepcopy(delta_new)
        return [sigma, GS, True]

    ''' Chase and remove conflict temporal orders in deltaTOML
    '''
    def ChaseAndRemoveTOML(self, CCs, GS, deltaTOStable, deltaTOML, indicator_last, D, attrsMap, round):
        fixes = Fixes()
        sigma = []
        if indicator_last == False:
            for to in deltaTOStable:
                sigma += GS.retrieveAllRelatedTOs(to)
        # add TOs to fixes
        for to in sigma:
            fixes.insert(to)
        delta = deltaTOStable + sigma
        if round <= 0:
            # transtivity
            delta_new = []
            for to in delta:
                res = self.transitivityAndRemoveTOML(to, GS, fixes, deltaTOML, sigma, delta_new)
                if res != None:
                    return res
            delta += delta_new
        while len(delta) > 0:
            delta_new = []
            for to in delta:
                newH = self.envokeTOLHS(to, CCs, GS)
                GS.update(newH)
                l_start, l_list = GS.findValuationsByTO(to)
                for i in range(l_start, len(l_list), 1):
                    vid = l_list[i]
                    # get the valuation
                    h = GS.getValuation(vid)
                    # check the RHS TO of h has been handled and placed in the stable set
                    if GS.isStableTO(h.getRHSTO()):
                        continue
                    # mark to as validate in h
                    h.addValidTO(to)
                    if h.isComplete():
                        # when complete, deduce RHS temporal order
                        to_rhs = h.deduceRHS(attrsMap, D, fixes)
                        GS.remove(vid, i)
                        # check conflict
                        if fixes.checkTemporalOrder_(to_rhs) == Status.FAILURE: # or to_rhs.encodeInverse() in deltaTOML:   # conflict
                            print("Conflict I")
                            return [[to_rhs], GS, False]
                        if to_rhs.encodeInverse() in deltaTOML:     # conflict with TO deduced by ML, do not report "CONFLICT" but remove them from deltaTOML
                            deltaTOML.remove(to_rhs.encodeInverse())
                        # transitivity 1.0
                        '''
                        tos_trans_res = []
                        tos_trans = GS.generateTransitivity(to_rhs)
                        for to_trans in tos_trans:
                            if fixes.checkTemporalOrder_(to_trans) == Status.FAILURE or to_trans.encodeInverse() in deltaTOML: #conflict
                                return [[to_trans], GS, False]
                            tos_trans_res.append(to_trans)
                        tos_trans = fixes.generateTransitivity(to_rhs)
                        for to_trans in tos_trans:
                            if GS.isStableTO(to_trans):
                                continue
                            if fixes.checkTemporalOrder_(to_trans) == Status.FAILURE or to_trans.encodeInverse() in deltaTOML: # conflict
                                return [[to_trans], GS, False]
                            tos_trans_res.append(to_trans)

                        fixes.insert(to_rhs)
                        sigma.append(to_rhs)
                        delta_new.append(to_rhs)

                        # add transitivity temporal orders
                        for to_trans in tos_trans_res:
                            fixes.insert(to_trans)
                            sigma.append(to_trans)
                            delta_new.append(to_trans)
                        '''
                        # transitivity 2.0
                        res = self.transitivityAndRemoveTOML(to_rhs, GS, fixes, deltaTOML, sigma, delta_new)
                        if res != None:
                            return res

            delta = copy.deepcopy(delta_new)
        return [sigma, GS, True]


    def transitivity(self, to_rhs, GS, fixes, deltaTOML, sigma, delta_new):
        fixes.insert(to_rhs)
        sigma.append(to_rhs)
        delta_new.append(to_rhs)
        tos_trans = [to_rhs]
        while len(tos_trans) > 0:
            res = []
            for to in tos_trans:
                tos_trans_ = GS.generateTransitivity(to)
                for to_ in tos_trans_:
                    if fixes.checkTemporalOrder_(to_) == Status.FAILURE or to_.encodeInverse() in deltaTOML:  # conflict
                        return [[to_], GS, False]
                    if fixes.checkTemporalOrder_(to_) == Status.SUCCESS:
                        continue
                    fixes.insert(to_)
                    sigma.append(to_)
                    delta_new.append(to_)
                    res.append(to_)
                tos_trains_ = fixes.generateTransitivity(to)
                for to_ in tos_trains_:
                    if GS.isStableTO(to_):
                        continue
                    if fixes.checkTemporalOrder_(to_) == Status.FAILURE or to_.encodeInverse() in deltaTOML:  # conflict
                        return [[to_], GS, False]
                    if fixes.checkTemporalOrder_(to_) == Status.SUCCESS:
                        continue
                    fixes.insert(to_)
                    sigma.append(to_)
                    delta_new.append(to_)
                    res.append(to_)
            tos_trans = res
        return None


    def transitivityAndRemoveTOML(self, to_rhs, GS, fixes, deltaTOML, sigma, delta_new):
        fixes.insert(to_rhs)
        sigma.append(to_rhs)
        delta_new.append(to_rhs)
        tos_trans = [to_rhs]
        while len(tos_trans) > 0:
            res = []
            for to in tos_trans:
                tos_trans_ = GS.generateTransitivity(to)
                for to_ in tos_trans_:
                    if fixes.checkTemporalOrder_(to_) == Status.FAILURE: # or to_.encodeInverse() in deltaTOML:  # conflict
                        print("Conflict II")
                        continue
                        # return [[to_], GS, False]
                    if to_.encodeInverse() in deltaTOML:
                        deltaTOML.remove(to_.encodeInverse())
                    if fixes.checkTemporalOrder_(to_) == Status.SUCCESS:
                        continue
                    fixes.insert(to_)
                    sigma.append(to_)
                    delta_new.append(to_)
                    res.append(to_)
                tos_trains_ = fixes.generateTransitivity(to)
                for to_ in tos_trains_:
                    if GS.isStableTO(to_):
                        continue
                    if fixes.checkTemporalOrder_(to_) == Status.FAILURE: # or to_.encodeInverse() in deltaTOML:  # conflict
                        print("Conflict III")
                        continue
                        # return [[to_], GS, False]
                    if to_.encodeInverse() in deltaTOML:
                        deltaTOML.remove(to_.encodeInverse())
                    if fixes.checkTemporalOrder_(to_) == Status.SUCCESS:
                        continue
                    fixes.insert(to_)
                    sigma.append(to_)
                    delta_new.append(to_)
                    res.append(to_)
            tos_trans = res
        return None



    ''' a naive version of chase, i.e., GATENC
    '''

    def envokeTOLHSNC(self, to, ccs, GS):
        valuations = []
        eid = to.getEID()
        attr = to.getAttr()
        v_1, v_2 = self.D[to.t0][self.attrMap[attr]], self.D[to.t1][self.attrMap[attr]]
        ccsIDList = ccs.getIndex()[attr]
        if ccsIDList == None:
            return valuations
        for cc_id in ccsIDList:
            CC = ccs.getCC(cc_id)
            t_0, t_1 = to.t0, to.t1
            v = Valuation(CC, cc_id, t_0, t_1, eid, [])
            key_v = v.encode()
            # check whether the valudation has been computed
            #if GS.isComputedV(key_v):
                #continue
            # check whether a TO is not correct
            if not v.checkTOPredicates_(GS):
                continue
            if not v.checkNonTOPredicates(self.attrMap, self.D):
                continue
            valuations.append(v)
        return valuations



    ''' assume rules whose LHSs do not contain temporal orders are filtered
        deltaTOStable is the new stable list of temporal orders
        deltaTOML is the defaultdict, where the key is the encode and value is TO
    '''
    def ChaseNC(self, CCs, GS, deltaTOStable, deltaTOML, indicator_last, D, attrsMap):
        fixes = Fixes()
        sigma = []
        if indicator_last == False:
            for to in deltaTOStable:
                sigma += GS.retrieveAllRelatedTOs(to)
        # add TOs to fixes
        for to in sigma:
            fixes.insert(to)
        delta = deltaTOStable + sigma
        round = 0
        while len(delta) > 0:
            round = 0
            delta_new = []
            for to in delta:
                newH = self.envokeTOLHS(to, CCs, GS)
                GS.update(newH)
                if round <= 20:
                    l_start, l_list = GS.findValuationsByTONC(to) #GS.findValuationsByTO(to)
                else:
                    l_start, l_list = GS.findValuationsByTO(to)
                round += 1
                for i in range(l_start, len(l_list), 1):
                    vid = l_list[i]
                    # get the valuation
                    h = GS.getValuation(vid)
                    if h == None:
                        continue
                    # if not h.ifContainTONC(to):
                        # continue

                    # check the RHS TO of h has been handled and placed in the stable set
                    if GS.isStableTO(h.getRHSTO()):
                        continue
                    # mark to as validate in h
                    FF = h.addValidTONC(to)
                    if not FF:
                        continue
                    if h.isComplete():
                        # when complete, deduce RHS temporal order
                        to_rhs = h.deduceRHS(attrsMap, D, fixes)
                        GS.removeNC(vid, i)
                        # check conflict
                        if fixes.checkTemporalOrder_(to_rhs) == Status.FAILURE or to_rhs.encodeInverse() in deltaTOML:   # conflict
                            return [[to_rhs], GS, False]
                        # transitivity 1.0
                        '''
                        tos_trans_res = []
                        tos_trans = GS.generateTransitivity(to_rhs)
                        for to_trans in tos_trans:
                            if fixes.checkTemporalOrder_(to_trans) == Status.FAILURE or to_trans.encodeInverse() in deltaTOML: #conflict
                                return [[to_trans], GS, False]
                            tos_trans_res.append(to_trans)
                        tos_trans = fixes.generateTransitivity(to_rhs)
                        for to_trans in tos_trans:
                            if GS.isStableTO(to_trans):
                                continue
                            if fixes.checkTemporalOrder_(to_trans) == Status.FAILURE or to_trans.encodeInverse() in deltaTOML: # conflict
                                return [[to_trans], GS, False]
                            tos_trans_res.append(to_trans)

                        fixes.insert(to_rhs)
                        sigma.append(to_rhs)
                        delta_new.append(to_rhs)

                        # add transitivity temporal orders
                        for to_trans in tos_trans_res:
                            fixes.insert(to_trans)
                            sigma.append(to_trans)
                            delta_new.append(to_trans)
                        '''
                        # transitivity 2.0
                        res = self.transitivity(to_rhs, GS, fixes, deltaTOML, sigma, delta_new)
                        if res != None:
                            return res

            delta = copy.deepcopy(delta_new)
        return [sigma, GS, True]



    # def ChaseNC(self, CCs, GS, deltaTOStable, deltaTOML, indicator_last, D, attrsMap):
    #     fixes = Fixes()
    #     sigma = []
    #     if indicator_last == False:
    #         for to in deltaTOStable:
    #             sigma += GS.retrieveAllRelatedTOs(to)
    #     # add TOs to fixes
    #     for to in sigma:
    #         fixes.insert(to)
    #     delta = deltaTOStable + sigma
    #     while len(delta) > 0:
    #         delta_new = []
    #         for to in delta:
    #             newH = self.envokeTOLHS(to, CCs, GS)
    #             for h in newH:
    #                 if GS.isStableTO(h.getRHSTO()):
    #                     continue
    #                 h.addValidTO(to)
    #                 if h.isComplete():
    #                     # when complete, deduce RHS temporal order
    #                     to_rhs = h.deduceRHS(attrsMap, D, fixes)
    #                     #GS.remove(vid, i)
    #                     # check conflict
    #                     if fixes.checkTemporalOrder_(to_rhs) == Status.FAILURE or to_rhs.encodeInverse() in deltaTOML:   # conflict
    #                         return [[to_rhs], GS, False]
    #                     # transitivity
    #                     tos_trans_res = []
    #                     tos_trans = GS.generateTransitivity(to_rhs)
    #                     for to_trans in tos_trans:
    #                         if fixes.checkTemporalOrder_(to_trans) == Status.FAILURE or to_trans.encodeInverse() in deltaTOML: #conflict
    #                             return [[to_trans], GS, False]
    #                         tos_trans_res.append(to_trans)
    #                     tos_trans = fixes.generateTransitivity(to_rhs)
    #                     for to_trans in tos_trans:
    #                         if GS.isStableTO(to_trans):
    #                             continue
    #                         if fixes.checkTemporalOrder_(to_trans) == Status.FAILURE or to_trans.encodeInverse() in deltaTOML: # conflict
    #                             return [[to_trans], GS, False]
    #                         tos_trans_res.append(to_trans)

    #                     fixes.insert(to_rhs)
    #                     sigma.append(to_rhs)
    #                     delta_new.append(to_rhs)

    #                     # add transitivity temporal orders
    #                     for to_trans in tos_trans_res:
    #                         fixes.insert(to_trans)
    #                         sigma.append(to_trans)
    #                         delta_new.append(to_trans)

    #         delta = copy.deepcopy(delta_new)
    #     return [sigma, GS, True]

    '''
    def BChase(self, ccsArr, fixes):
        ccs_0, ccs_1 = self.classifyCCs(ccsArr)
        TOs = self.execute(ccs_1, fixes)
        valuations = self.envokeTOLHS(ccs_0, fixes)
        # index valuations
        indexValuations_part, indexValuations_comp = defaultdict(list), defaultdict(list)
        valuations_part, valuations_comp = [], []
        for valuation in valuations:
            if valuation.isComplete():
                valuations_comp.append(valuation)
            else:
                valuations_part.append(valuation)

        for vid, valuation in enumerate(valuations_part):
            for k in valuation.getTOKeys():
                indexValuations_part[k].append(vid)
        for vid, valuation in enumerate(valuations_comp):
            for k in valuation.getTOKeys():
                indexValuations_comp[k].append(vid)

        partialResults = PartialResults(valuations_part, valuations_comp, indexValuations_part, indexValuations_comp, fixes, TOs)
        return partialResults
    '''

    '''
    def IncChase(self, partialResults):
        valuation_part_ = copy.deepcopy(partialResults.valuations_part)
        valuation_comp_ = copy.deepcopy(partialResults.valuation_comp)
        indexValuations_part = copy.deepcopy(partialResults.indexValuations_part)
        indexValuations_comp = copy.deepcopy(partialResults.indexValuations_comp)
        fixes_ = copy.deepcopy(fixes)
        # 1. incremental deduction in partial valuations
        for to in partialResults.deltaTOs:
    '''










