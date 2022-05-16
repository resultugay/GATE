import itertools


class Critic:
    def __init__(self):
        pass


    def deduce(self, high_conf_css, training_data, ccs):

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
