from collections import defaultdict
from Valuation import *

class Chase(object):


    ''' the first column is TID, the second column is EID
    '''
    def __init__(self, D, schemas):
        self.D = D # original dataset (only consider ONE)
        self.schemas = schemas
        self.PLIs = []
        self.computedValuations = defaultdict()

    def constructPLIs(self):
        for cid in range(2, len(self.schemas), 1):
            pli = self.constructPLI(D, cid)
            self.PLIs.append(pli)

    def constructPLI(self, D, cid):
        pli = defaultdict(list)
        for tid in range(len(D)):
            c = D[tid][cid]
            pli[c].append(tid)

    def envoke(self, ccs, fixes):



    def PChase(self, ccs, fixes):



