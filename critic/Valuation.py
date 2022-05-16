from Fixes import *
RELATION_ATTRIBUTE = "___"
PREDICATE_DELIMITOR = "^"


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
            a_1 = self.operand1['relation'] + RELATION_ATTRIBUTE + self.operand1['attribute']
            a_2 = self.operand2['relation'] + RELATION_ATTRIBUTE + self.operand2['attribute']
            if a_1 in timelinessAttrs and a_2 in timelinessAttrs:
                self.isTimeliness = True

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

    ''' do not consider ML
    '''
    def ifSatisfy(self, t0, t1, eid, attrMap, fixes, selectedTOs=None):
        if self.index1 == self.index2:
            # constant predicate
            attrID = attrMap[self.operand1['attribute']]
            if t0[attrID] == self.constant:
                return Status.SUCCESS
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
                    return fixes.checkTemporalOrder(t1[attrID_2], t0[attrID_1], eid, self.operand1['attribute'], selectedTOs)
                elif self.operator == '<':
                    return fixes.checkTemporalOrder(t0[attrID_1], t1[attrID_2], eid, self.operand1['attribute'], selectedTOs)
                else:
                    print('Error predicate !!!')
        return Status.FAILURE

class CC(object):
    def __init__(self, ruleStr, timelinessAttrs):
        self.lhs, self.rhs = [], None
        ruleStrSet = [e.strip() for e in ruleStr.split(PREDICATE_DELIMITOR)]
        self.lhs = [Predicate(e.strip(), timelinessAttrs) for e in ruleStrSet[:-1]]
        self.rhs = Predicate(ruleStrSet[-1], timelinessAttrs)
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

class CCs(object):
    def __init__(self, ruleStrs, timelinessAttrs):
        self.CCsList = []
        for ruleStr in ruleStrs:
            self.CCsList.append(CC(ruleStr, timelinessAttrs))

        self.indexByTO()

    def getCC(self, rid):
        return self.CCsList[rid]

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
            if predicate.ifSatisfy(D[self.t0], D[self.t1], attrMap, None) == Status.FAILURE:
                return False
        return True

    def checkTOPredicates(self, attrMap, D, fixes):
        unselected = [pid for pid in range(len(self.CC.getLHSs())) if pid not in self.satistiedLHSs]
        for pid in unselected:
            predicate = self.CC.getLHSs()[pid]
            if not predicate.isTimelinessPredicate():
                continue
            status = predicate.ifSatisfy(D[self.t0], D[self.t1], attrMap, fixes, self.TOKeys)
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
        v_1, v_2 = D[self.t0][attrMap[attr_1]], D[self.t1][attrMap[attr_2]]
        conf = fixes.computeConfTOs(self.TOKeys)
        to = fixes.generateAndSaveTO(v_1, v_2, attr_1, op, self.eid, conf)
        return to





