import logging

import Gate
from .Critic import Critic
import itertools

class MeasuresCritic(Critic):
    def __init__(self, args):
        self.args = args
        logging.info('Critic is measures')

    def choose_rules(self, currency_constraints, df):
        return_remove_list = {}

        for col, rules in currency_constraints.items():
            remove_list = {}
            for id_ in df.index.unique():
                sub_df = df.loc[id_]
                sub_df.index = sub_df.row_id
                rows = sub_df.to_dict('Records')
                for i in itertools.combinations(rows, 2):
                    if i[0]['timestamp'] == i[1]['timestamp']:
                        continue
                    value_tuple_1 = i[0][col]
                    value_tuple_2 = i[1][col]

                    for val1, val2 in rules:
                        if (val1, val2) not in remove_list:
                            remove_list[(val1, val2)] = 0
                        if i[0]['timestamp'] > i[1]['timestamp']:
                            if val1 == value_tuple_1 and val2 == value_tuple_2:
                                remove_list[(val1,val2)] += 1
                            elif val1 == value_tuple_2 and val2 == value_tuple_1:
                                remove_list[(val1, val2)] -= 1

                        elif i[0]['timestamp'] < i[1]['timestamp']:
                            if val1 == value_tuple_1 and val2 == value_tuple_2:
                                remove_list[(val1,val2)] -= 1
                            elif val1 == value_tuple_2 and val2 == value_tuple_1:
                                remove_list[(val1, val2)] += 1
            remove_list = sorted(remove_list.items(), key=lambda x: x[1])
            remove_list = [x[0] for x in remove_list[:round(len(remove_list) * 0.3)]]
            return_remove_list[col] = remove_list


        return return_remove_list