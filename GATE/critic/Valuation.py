from .Fixes import *
RELATION_ATTRIBUTE = "___"
PREDICATE_DELIMITOR = "^"
LHS_RHS_DELIMITOR = '->'


class Predicate(object):
    def __init__(self, predicate_str, timelinessAttrs):
        self.index1 = None
        self.index2 = None
        self.operator = None
        self.operand1 = None
        self.operand2 = None
        self.constant = None
        self.isTimeliness = False
        res = self.parsePredicate(predicate_str)
        self.predicateStr = predicate_str
        self.index1 = res[0]
        self.operand1 = {'relation': res[1].split(RELATION_ATTRIBUTE)[0].strip(), 'attribute': res[1].split(RELATION_ATTRIBUTE)[1].strip()}
        self.operator = res[2]
        self.index2 = res[3]
        if self.index1 == self.index2:
            self.constant = res[4]
        else:
            self.operand2 = {'relation': res[4].split(RELATION_ATTRIBUTE)[0].strip(), 'attribute': res[4].split(RELATION_ATTRIBUTE)[1].strip()}
            #a_1 = self.operand1['relation'] + RELATION_ATTRIBUTE + self.operand1['attribute']
            a_1 = self.operand1['attribute']
            #a_2 = self.operand2['relation'] + RELATION_ATTRIBUTE + self.operand2['attribute']
            a_2 = self.operand2['attribute']
            if a_1 in timelinessAttrs and a_2 in timelinessAttrs:
                self.isTimeliness = True

    def extractPredicateTuplePairsList(self, D_groupbyEIDs, index, attrsMap):
        tps = defaultdict(list)
        # 1. t.A = s.B form
        if self.operator == '==' and self.constant == None:
            for k, vdict in index.items():
                for eid, bucket in vdict.items():
                    for tid_0 in bucket:
                        for tid_1 in bucket:
                            if tid_0 == tid_1:
                                continue
                            tps[eid].append([tid_0, tid_1])
        # 2. t.A = c form
        elif self.operator == '==' and self.constant != None:
            vdict = index[self.constant]
            if vdict != None:
                for eid, bucket in vdict.items():
                    # make pairs with tuples in the same EIDs
                    for tid0 in bucket:
                        D_eid = D_groupbyEIDs[eid]
                        for t in D_eid:
                            tid1 = t[0]
                            if tid0 == tid1:
                                continue
                            tps[eid].append([tid0, tid1])
        # 3. t.A < s.B
        else:
            attr_1, attr_2 = attrsMap[self.operand1['attribute']], attrsMap['attribute']
            for eid, D_eid in D_groupbyEIDs.items():
                for t0 in D_eid:
                    tid_0 = t0[0]
                    for t1 in D_eid:
                        tid_1 = t1[0]
                        if tid_0 == tid_1:
                            continue
                        v_1, v_2 = t0[attr_1], t1[attr_2]
                        if self.operator == '<' and v_1 < v_2:
                            tps[eid].append([tid_0, tid_1])
                        elif self.operator == '<=' and v_1 <= v_2:
                            tps[eid].append([tid_0, tid_1])
                        elif self.operator == ">" and v_1 > v_2:
                            tps[eid].append([tid_0, tid_1])
                        elif self.operator == ">=" and v_1 >= v_2:
                            tps[eid].append([tid_0, tid_1])
        return tps

    def toString(self):
        # constant predicates
        if self.constant != None:
            return self.index1 + '.' + self.operand1['relation'] + '.' + self.operand1[
                'attribute'] + ' ' + self.operator + ' ' + self.constant
        else:
            return self.index1 + '.' + self.operand1['relation'] + '.' + self.operand1[
                'attribute'] + ' ' + self.operator + ' ' + self.index2 + '.' + self.operand2['relation'] + '.' + \
                   self.operand2['attribute']

    '''
    def toString(self):
        if self.constant != None:
            return self.operand1['relation'] + '.' + self.operand1['attribute'] + '.' + 
    '''

    def isTimelinessPredicate(self):
        return self.isTimeliness

    def getAttrs(self):
        return self.operand1['attribute'], self.operand2['attribute']

    def getOp(self):
        return self.operator

    def isConstant(self):
        if self.constant == None:
            return False
        return True

    def parsePredicate(self, predicate):
        # check Relation(t0)  ---- 提取规则中的 t0 t1这样的 比如： rule=casorgcn(t0) rule=order(t1)  提取的都是括号里面的 t0 t1
        if predicate.find("(") != -1 and predicate.find(")") != -1 and predicate[:2] != 'ML' and predicate[:len(
                'similar')] != 'similar':
            ss = predicate[predicate.find("(") + 1:predicate.find(")")]
            for i in range(1, len(ss), 1):
                if ss[:i].isalpha and ss[i:].isdigit:
                    return None

        res = None
        if predicate[:2] == 'ML':
            t = predicate.split('(')
            operator = t[0]
            op = predicate[predicate.find('(') + 1: predicate.find(")")]
            operand1, operand2 = op.split(',')[0].strip(), op.split(',')[1].strip()
            res = [operand1, operator, operand2]
        elif len(predicate) >= len('similar') and predicate[:len('similar')] == 'similar':
            ss = [e.strip() for e in predicate[len('similar') + 1: -1].split(",")]
            op = ss[0] + ' ' + ss[-1]
            res = [ss[1], op, ss[2]]
        else:
            t = predicate.split()
            operand1 = t[0]
            operator = t[1]
            operand2 = ' '.join(t[2:])
            res = [operand1, operator, operand2]

        operand1_ = res[0].split(".")
        index1_ = operand1_[1]
        operand1_new = operand1_[0] + RELATION_ATTRIBUTE + operand1_[2]
        operator_ = res[1]
        operand2_ = res[2].split(".")
        if len(operand2_) < 3:
            index2_ = index1_
            operand2_new = res[2]
        else:
            index2_ = operand2_[1]
            operand2_new = operand2_[0] + RELATION_ATTRIBUTE + operand2_[2]

        return [index1_, operand1_new, operator_, index2_, operand2_new]

    def transformTO(self, t0, t1, eid):
        '''
        Only used for predicate of temporal order
        :param t0:
        :param t1:
        :param eid:
        :return:
        '''
        op = Operator.GREATOR_THAN
        if self.operator == "<":
            op = Operator.LESS_THAN
        to = TemporalOrder(t0, t1, self.operand1['attribute'], op, eid, 1.0)
        return to

    ''' do not consider ML
    '''
    def ifSatisfy(self, tid0, tid1, t0, t1, eid, attrMap, fixes, selectedTOs=None):
        if self.index1 == self.index2 and self.constant != None:
            # constant predicate
            attrID = attrMap[self.operand1['attribute']]
            if self.operator == '==':
                if str(t0[attrID]) == str(self.constant):
                    return Status.SUCCESS
            elif self.operator == '<=':
                if t0[attrID] <= float(self.constant):
                    return Status.SUCCESS
            elif self.operator == '<':
                if t0[attrID] < float(self.constant):
                    return Status.SUCCESS
            elif self.operator == '>=':
                if t0[attrID] >= float(self.constant):
                    return Status.SUCCESS
            elif self.operator == '>':
                if t0[attrID] > float(self.constant):
                    return Status.SUCCESS
            else:
                print('Error predicate !!!')
        elif self.operator == '==':
            attrID_1, attrID_2 = attrMap[self.operand1['attribute']], attrMap[self.operand2['attribute']]
            if t0[attrID_1] == t1[attrID_2]:
                return Status.SUCCESS
            else:
                return Status.FAILURE
        else:
            attrID_1, attrID_2 = attrMap[self.operand1['attribute']], attrMap[self.operand2['attribute']]
            if self.isTimeliness == False:
                if self.operator == ">":
                    if t0[attrID_1] > t1[attrID_2]:
                        return Status.SUCCESS
                elif self.operator == "<":
                    if t0[attrID_1] < t1[attrID_2]:
                        return Status.SUCCESS
                elif self.operator == '>=':
                    if t0[attrID_1] >= t1[attrID_2]:
                        return Status.SUCCESS
                elif self.operator == '<=':
                    if t0[attrID_1] <= t1[attrID_2]:
                        return Status.SUCCESS
                else:
                    print('Error predicate !!!')
            else:
                if self.operator == '>':
                    return fixes.checkTemporalOrder(tid1, tid0, eid, self.operand1['attribute'], selectedTOs)
                elif self.operator == '<':
                    return fixes.checkTemporalOrder(tid0, tid1, eid, self.operand1['attribute'], selectedTOs)
                else:
                    print('Error predicate !!!')
        return Status.FAILURE

class CC(object):
    def __init__(self, ruleStr, timelinessAttrs):
        self.lhs, self.rhs = [], None
        lhsStr, rhsStr = ruleStr.split(LHS_RHS_DELIMITOR)[0].strip(), ruleStr.split(LHS_RHS_DELIMITOR)[1].strip()
        lhsStrSet = [e.strip() for e in lhsStr.split(PREDICATE_DELIMITOR)]
        self.lhs = [Predicate(e.strip(), timelinessAttrs) for e in lhsStrSet]
        self.rhs = Predicate(rhsStr, timelinessAttrs)
        self.timelinessPIDs = []
        for pid, p in enumerate(self.lhs):
            if p.isTimelinessPredicate():
                self.timelinessPIDs.append(pid)

    def getLHSs(self):
        return self.lhs

    def getRHS(self):
        return self.rhs

    def getTimelinessPIDs(self):
        return self.timelinessPIDs

    def getTPredicateInLHS(self, attr):
        for pid in self.timelinessPIDs:
            p = self.lhs[pid]
            attr_1, attr_2 = p.getAttrs()
            if attr == attr_1:
                return p
        return None

    ''' only for CCs whose LHSs do not contain timeliness and RHS is a timeliness predicate
    '''
    def getOneNonTOPredicate(self):
        # 1. extract "=" predicate
        eqList = []
        for p in self.lhs:
            if p.getOp() == "==" and not p.isConstant():
                eqList.append(p)
                break
        # 2. if not t.A = s.B, check t.A = c and s.B = d
        if len(eqList) == 0:
            for p in self.lhs:
                if p.getOp == "==" and p.isConstant():
                    eqList.append(p)
                    break

        # 3. the remaining one case: t.A > s.B or t.A < s.B, where A, B are numeric attributes
        if len(eqList) == 0:
            for p in self.lhs:
                eqList.append(p)
                break
        return eqList


class CCs(object):
    def __init__(self, ruleStrs, timelinessAttrs):
        self.CCsList = []
        for ruleStr in ruleStrs:
            self.CCsList.append(CC(ruleStr, timelinessAttrs))
        self.indexByTO()

    def getCC(self, rid):
        return self.CCsList[rid]

    def getIndex(self):
        return self.CCsIndex

    def indexByTO(self):
        self.CCsIndex = defaultdict(list)
        for cc_id, cc in enumerate(self.CCsList):
            for p in cc.getLHSs():
                if p.isTimelinessPredicate():
                    attr1, attr2 = p.getAttrs()
                    if attr1 not in self.CCsIndex or self.CCsIndex[attr1][-1] != cc_id:
                        self.CCsIndex[attr1].append(cc_id)

class Valuation(object):
    def __init__(self, CC, cc_id, t_0, t_1, eid, satisfiedLHSs):
        self.CC = CC
        self.cc_id = cc_id
        self.t0 = t_0
        self.t1 = t_1
        self.eid = eid
        self.TOKeys = []
        self.satistiedLHSs = [e for e in satisfiedLHSs]
        # RHS temporal order
        self.to_rhs = None
        # set temporal orders
        self.toPredicates = defaultdict()  # key is the key of TO, value is [temporal order, id of predicates in CC]
        for id_ in self.CC.getTimelinessPIDs():
            to = self.CC.getLHSs()[id_].transformTO(t_0, t_1, eid)
            k = to.encode()
            self.toPredicates[k] = [to, id_]
            self.TOKeys.append(k)
        # temporal order of rhs
        self.to_rhs = self.CC.getRHS().transformTO(t_0, t_1, eid)
        # validated TOs
        self.validTOs = []

    def setTOKeys(self, toKeys):
        self.TOKeys = toKeys

    def getTOKeys(self):
        return self.TOKeys

    def addValidTO(self, to):
        self.validTOs.append(to)
        pid = self.toPredicates[to.encode()][1]
        self.satistiedLHSs.append(pid)

    def addValidTONC(self, to):
        if to.encode() in self.toPredicates:
            self.validTOs.append(to)
            pid = self.toPredicates[to.encode()][1]
            self.satistiedLHSs.append(pid)
            return True
        else:
            return False

    def ifContainTONC(self, to):
        return to.encode() in self.toPredicates

    def isComplete(self):
        return len(self.satistiedLHSs) == len(self.CC.getLHSs())

    def encode(self):
        return str(self.t0) + RELATION_ATTRIBUTE + str(self.t1) + RELATION_ATTRIBUTE + str(self.cc_id)

    def checkNonTOPredicates(self, attrMap, D):
        # 1. get all non-to predicates
        unselected = [pid for pid in range(len(self.CC.getLHSs())) if pid not in self.satistiedLHSs]
        # 2. check all non-to predicates
        for pid in unselected:
            predicate = self.CC.getLHSs()[pid]
            if predicate.isTimelinessPredicate():
                continue
            if predicate.ifSatisfy(self.t0, self.t1, D[self.t0], D[self.t1], attrMap, None) == Status.FAILURE:
                return False
        return True

    def checkTOPredicates_(self, GS):
        for k, v in self.toPredicates.items():
            to = v[0]
            if GS.isValidTO(to) == False:
                return False
        return True

    def checkTOPredicates(self, attrMap, D, fixes):
        unselected = [pid for pid in range(len(self.CC.getLHSs())) if pid not in self.satistiedLHSs]
        for pid in unselected:
            predicate = self.CC.getLHSs()[pid]
            if not predicate.isTimelinessPredicate():
                continue
            status = predicate.ifSatisfy(self.t0, self.t1, D[self.t0], D[self.t1], attrMap, fixes)
            if status == Status.SUCCESS:
                self.satistiedLHSs.append(pid)
                continue
            elif status == Status.FAILURE:
                return False
            else:
                continue
        return True

    def deduceRHS(self, attrMap, D, fixes):
        attr_1, attr_2 = self.CC.getRHS().getAttrs()
        op = self.CC.getRHS().getOp()
        #v_1, v_2 = D[self.t0][attrMap[attr_1]], D[self.t1][attrMap[attr_2]]
        t_0, t_1 = self.t0, self.t1
        conf = 1.0 #fixes.computeConfTOs(self.TOKeys)
        op_ = Operator.LESS_THAN
        if op == '>':
            op_ = Operator.GREATOR_THAN
        elif op == '<':
            op_ = Operator.LESS_THAN
        to = fixes.generateAndSaveTO(t_0, t_1, attr_1, op_, self.eid, conf)
        self.to_rhs = to
        return to

    def getRHSTO(self):
        return self.to_rhs


