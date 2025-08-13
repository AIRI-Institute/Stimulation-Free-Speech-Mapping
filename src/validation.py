import numpy as np


def get_table(df, cns, esm_negative, esm_positive):
    table = {
        'p+|a+': 0,
        'p-|a+': 0,
        'p+|a-': 0,
        'p-|a-': 0,
    }
    
    for index, row in df.iterrows():
        channel_name1, channel_name2, active = row
        if '-' in [channel_name1, channel_name2]:
            continue
        elif str(active) in esm_positive:
            if (channel_name1 in cns) or (channel_name2 in cns):
                table['p+|a+'] += 1
            elif (channel_name1 not in cns) and (channel_name2 not in cns):
                table['p-|a+'] += 1
        elif str(active) in esm_negative:
            if (channel_name1 not in cns) and (channel_name2 not in cns):
                table['p-|a-'] += 1
                continue
            if channel_name1 in cns:
                if len(esm_negative.intersection(set(df[df['Channel 2'] == channel_name1]['Result'].values))) != 0:
                    table['p+|a-'] += 1
                    continue
            if channel_name2 in cns:
                if len(esm_negative.intersection(set(df[df['Channel 1'] == channel_name2]['Result'].values))) != 0:
                    table['p+|a-'] += 1
                    continue
    return table


def get_tables_subject(df, thresholds, subject_names, dfs_active, negative, positive):
    df_roc = df.copy()
    tables = {subject_name: [] for subject_name in subject_names}
    
    for subject_name in subject_names:
        df_roc_subject = df_roc.copy()
        df_roc_subject = df_roc_subject[df_roc_subject['subject_name'] == subject_name]
        df_active = dfs_active[subject_name]
        
        for i, thr in enumerate(thresholds):
            passive_positive = set(df_roc_subject.loc[df_roc_subject['prediction_proba'] >= thr, 'channel_name'].values)
            table = get_table(df_active, passive_positive, negative, positive)
            tables[subject_name].append(table)
    return tables


def get_statistic(table):
    sensitivity, specificity, f1score, precision, fpr = None, None, None, None, None
    if (table['p+|a+'] + table['p-|a+']) != 0:
        sensitivity = table['p+|a+'] / (table['p+|a+'] + table['p-|a+'])
    if (table['p-|a-'] + table['p+|a-']) != 0:
        specificity = table['p-|a-'] / (table['p-|a-'] + table['p+|a-'])
    if (2 * table['p+|a+'] + table['p+|a-'] + table['p-|a+']) != 0:
        f1score = 2 * table['p+|a+'] / (2 * table['p+|a+'] + table['p+|a-'] + table['p-|a+'])
    if (table['p+|a+'] + table['p+|a-']) != 0:
        precision = table['p+|a+'] / (table['p+|a+'] + table['p+|a-'])
    if (table['p+|a-'] + table['p-|a-']) != 0:
        fpr = table['p+|a-'] / (table['p+|a-'] + table['p-|a-'])
    return sensitivity, specificity, f1score, precision, fpr


def get_average_roc(tables, thresholds, subject_names):
    tpr = np.zeros(len(thresholds))
    fpr = np.zeros(len(thresholds))
    for i, thr in enumerate(thresholds):
        table_total = {
            'p+|a+': 0,
            'p-|a+': 0,
            'p+|a-': 0,
            'p-|a-': 0,
        }
    
        for subject_name in subject_names:
            table = tables[subject_name][i]
            for k, v in table.items():
                table_total[k] += v
    
        sensitivity, specificity, f1score, precision, falsepr = get_statistic(table_total)
        tpr[i] = sensitivity
        fpr[i] = falsepr
    return fpr, tpr
