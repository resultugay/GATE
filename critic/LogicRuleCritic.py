import logging
from .Critic import Critic


class LogicRuleCritic(Critic):
    def __init__(self, args):
        self.args = args
        self.CC = {}
        logging.info('Critic is Logic Rule')
        self.read_CC()

    def read_CC(self):
        with open('critic/cc.txt') as f:
            lines = f.readlines()
            for line in lines:
                all_statements = line.strip().split(',')
                col = all_statements[0]
                t1 = all_statements[1]
                t2 = all_statements[2]
                if col not in self.CC:
                    self.CC[col] = []
                self.CC[col].append((t1,t2))

    def choose_rules(self, currency_constraints):
        remove_list = {}

        for col, cc_1 in currency_constraints.items():
            if col not in remove_list:
                remove_list[col] = []
            cc_2 = self.CC[col]

            for l, nl in cc_1:
                if (nl, l) in cc_2:
                    remove_list[col].append((l, nl))

        return remove_list
