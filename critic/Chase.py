from collections import defaultdict
from Valuation import *

class Chase(object):

    def __init__(self, D, schemas):
        self.D = D # original dataset (only consider ONE)
        self.schemas = schemas
        self.PLIs = []
        self.

    def constructPLIs(self):
        for cid in range(len(self.schemas)):
            pli = self.constructPLI(D, cid)
            self.PLIs.append(pli)

    def constructPLI(self, D, cid):
        pli = defaultdict(list)
        for tid in range(len(D)):
            c = D[tid][cid]
            pli[c].append(tid)

    def PChase(self, ccs, fixes):


