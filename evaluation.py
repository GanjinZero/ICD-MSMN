from collections import defaultdict
import csv
import json
import numpy as np
import os
import sys
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from constant import MIMIC_2_DIR, MIMIC_3_DIR


def all_metrics(yhat, y, k=[5, 8, 15], yhat_raw=None, calc_auc=True):
    """
        Inputs:
            yhat: binary predictions matrix 
            y: binary ground truth matrix
            k: for @k metrics
            yhat_raw: prediction scores matrix (floats)
        Outputs:
            dict holding relevant metrics
    """
    names = ["acc", "prec", "rec", "f1"]

    #macro
    macro = all_macro(yhat, y)

    #micro
    ymic = y.ravel()
    yhatmic = yhat.ravel()
    micro = all_micro(yhatmic, ymic)

    metrics = {names[i] + "_macro": macro[i] for i in range(len(macro))}
    metrics.update({names[i] + "_micro": micro[i] for i in range(len(micro))})

    #AUC and @k
    if yhat_raw is not None and calc_auc:
        #allow k to be passed as int or list
        if type(k) != list:
            k = [k]
        for k_i in k:
            if k_i > y.shape[1]:
                continue
            rec_at_k = recall_at_k(yhat_raw, y, k_i)
            metrics['rec_at_%d' % k_i] = rec_at_k
            prec_at_k = precision_at_k(yhat_raw, y, k_i)
            metrics['prec_at_%d' % k_i] = prec_at_k
            metrics['f1_at_%d' % k_i] = 2*(prec_at_k*rec_at_k)/(prec_at_k+rec_at_k)

        roc_auc = auc_metrics(yhat_raw, y, ymic)
        metrics.update(roc_auc)

    return metrics

def all_macro(yhat, y):
    return macro_accuracy(yhat, y), macro_precision(yhat, y), macro_recall(yhat, y), macro_f1(yhat, y)

def all_micro(yhatmic, ymic):
    return micro_accuracy(yhatmic, ymic), micro_precision(yhatmic, ymic), micro_recall(yhatmic, ymic), micro_f1(yhatmic, ymic)

#########################################################################
#MACRO METRICS: calculate metric for each label and average across labels
#########################################################################

def macro_accuracy(yhat, y):
    num = intersect_size(yhat, y, 0) / (union_size(yhat, y, 0) + 1e-10)
    return np.mean(num)

def macro_precision(yhat, y):
    num = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    return np.mean(num)

def macro_recall(yhat, y):
    num = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
    return np.mean(num)

def macro_f1(yhat, y):
    prec = macro_precision(yhat, y)
    rec = macro_recall(yhat, y)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1

###################
# INSTANCE-AVERAGED
###################

def inst_precision(yhat, y):
    num = intersect_size(yhat, y, 1) / yhat.sum(axis=1)
    #correct for divide-by-zeros
    num[np.isnan(num)] = 0.
    return np.mean(num)

def inst_recall(yhat, y):
    num = intersect_size(yhat, y, 1) / y.sum(axis=1)
    #correct for divide-by-zeros
    num[np.isnan(num)] = 0.
    return np.mean(num)

def inst_f1(yhat, y):
    prec = inst_precision(yhat, y)
    rec = inst_recall(yhat, y)
    f1 = 2*(prec*rec)/(prec+rec)
    return f1

##############
# AT-K
##############

def recall_at_k(yhat_raw, y, k):
    #num true labels in top k predictions / num true labels
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]

    #get recall at k for each example
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y[i,tk].sum()
        denom = y[i,:].sum()
        vals.append(num_true_in_top_k / float(denom))

    vals = np.array(vals)
    vals[np.isnan(vals)] = 0.

    return np.mean(vals)

def precision_at_k(yhat_raw, y, k):
    #num true labels in top k predictions / k
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]

    #get precision at k for each example
    vals = []
    for i, tk in enumerate(topk):
        if len(tk) > 0:
            num_true_in_top_k = y[i,tk].sum()
            denom = len(tk)
            vals.append(num_true_in_top_k / float(denom))

    return np.mean(vals)

##########################################################################
#MICRO METRICS: treat every prediction as an individual binary prediction
##########################################################################

def micro_accuracy(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / (1e-10 + union_size(yhatmic, ymic, 0))

def micro_precision(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / (1e-10 + yhatmic.sum(axis=0))

def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / (1e-10 + ymic.sum(axis=0))

def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1

def auc_metrics(yhat_raw, y, ymic):
    if yhat_raw.shape[0] <= 1:
        return
    fpr = {}
    tpr = {}
    roc_auc = {}
    #get AUC for each label individually
    relevant_labels = []
    auc_labels = {}
    for i in range(y.shape[1]):
        #only if there are true positives for this label
        if y[:,i].sum() > 0:
            fpr[i], tpr[i], _ = roc_curve(y[:,i], yhat_raw[:,i])
            if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                auc_score = auc(fpr[i], tpr[i])
                if not np.isnan(auc_score): 
                    auc_labels["auc_%d" % i] = auc_score
                    relevant_labels.append(i)

    #macro-AUC: just average the auc scores
    aucs = []
    for i in relevant_labels:
        aucs.append(auc_labels['auc_%d' % i])
    roc_auc['auc_macro'] = np.mean(aucs)

    #micro-AUC: just look at each individual prediction
    yhatmic = yhat_raw.ravel()
    fpr["micro"], tpr["micro"], _ = roc_curve(ymic, yhatmic) 
    roc_auc["auc_micro"] = auc(fpr["micro"], tpr["micro"])

    return roc_auc

########################
# METRICS BY CODE TYPE
########################

def results_by_type(Y, mdir, version='mimic3'):
    d2ind = {}
    p2ind = {}

    #get predictions for diagnoses and procedures
    diag_preds = defaultdict(lambda: set([]))
    proc_preds = defaultdict(lambda: set([]))
    preds = defaultdict(lambda: set())
    with open('%s/preds_test.psv' % mdir, 'r') as f:
        r = csv.reader(f, delimiter='|')
        for row in r:
            if len(row) > 1:
                for code in row[1:]:
                    preds[row[0]].add(code)
                    if code != '':
                        try:
                            pos = code.index('.')
                            if pos == 3 or (code[0] == 'E' and pos == 4):
                                if code not in d2ind:
                                    d2ind[code] = len(d2ind)
                                diag_preds[row[0]].add(code)
                            elif pos == 2:
                                if code not in p2ind:
                                    p2ind[code] = len(p2ind)
                                proc_preds[row[0]].add(code)
                        except:
                            if len(code) == 3 or (code[0] == 'E' and len(code) == 4):
                                if code not in d2ind:
                                    d2ind[code] = len(d2ind)
                                diag_preds[row[0]].add(code)
    #get ground truth for diagnoses and procedures
    diag_golds = defaultdict(lambda: set([]))
    proc_golds = defaultdict(lambda: set([]))
    golds = defaultdict(lambda: set())
    test_file = '%s/test_%s.csv' % (MIMIC_3_DIR, str(Y)) if version == 'mimic3' else '%s/test.csv' % MIMIC_2_DIR
    with open(test_file, 'r') as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            codes = set([c for c in row[3].split(';')])
            for code in codes:
                golds[row[1]].add(code)
                try:
                    pos = code.index('.')
                    if pos == 3:
                        if code not in d2ind:
                            d2ind[code] = len(d2ind)
                        diag_golds[row[1]].add(code)
                    elif pos == 2:
                        if code not in p2ind:
                            p2ind[code] = len(p2ind)
                        proc_golds[row[1]].add(code)
                except:
                    if len(code) == 3 or (code[0] == 'E' and len(code) == 4):
                        if code not in d2ind:
                            d2ind[code] = len(d2ind)
                        diag_golds[row[1]].add(code)

    hadm_ids = sorted(set(diag_golds.keys()).intersection(set(diag_preds.keys())))

    ind2d = {i:d for d,i in d2ind.items()}
    ind2p = {i:p for p,i in p2ind.items()}
    type_dicts = (ind2d, ind2p)
    return diag_preds, diag_golds, proc_preds, proc_golds, golds, preds, hadm_ids, type_dicts


def diag_f1(diag_preds, diag_golds, ind2d, hadm_ids):
    num_labels = len(ind2d)
    yhat_diag = np.zeros((len(hadm_ids), num_labels))
    y_diag = np.zeros((len(hadm_ids), num_labels))
    for i,hadm_id in tqdm(enumerate(hadm_ids)):
        yhat_diag_inds = [1 if ind2d[j] in diag_preds[hadm_id] else 0 for j in range(num_labels)]
        gold_diag_inds = [1 if ind2d[j] in diag_golds[hadm_id] else 0 for j in range(num_labels)]
        yhat_diag[i] = yhat_diag_inds
        y_diag[i] = gold_diag_inds
    return micro_f1(yhat_diag.ravel(), y_diag.ravel())

def proc_f1(proc_preds, proc_golds, ind2p, hadm_ids):
    num_labels = len(ind2p)
    yhat_proc = np.zeros((len(hadm_ids), num_labels))
    y_proc = np.zeros((len(hadm_ids), num_labels))
    for i,hadm_id in tqdm(enumerate(hadm_ids)):
        yhat_proc_inds = [1 if ind2p[j] in proc_preds[hadm_id] else 0 for j in range(num_labels)]
        gold_proc_inds = [1 if ind2p[j] in proc_golds[hadm_id] else 0 for j in range(num_labels)]
        yhat_proc[i] = yhat_proc_inds
        y_proc[i] = gold_proc_inds
    return micro_f1(yhat_proc.ravel(), y_proc.ravel())

def metrics_from_dicts(preds, golds, mdir, ind2c):
    with open('%s/pred_100_scores_test.json' % mdir, 'r') as f:
        scors = json.load(f)

    hadm_ids = sorted(set(golds.keys()).intersection(set(preds.keys())))
    num_labels = len(ind2c)
    yhat = np.zeros((len(hadm_ids), num_labels))
    yhat_raw = np.zeros((len(hadm_ids), num_labels))
    y = np.zeros((len(hadm_ids), num_labels))
    for i,hadm_id in tqdm(enumerate(hadm_ids)):
        yhat_inds = [1 if ind2c[j] in preds[hadm_id] else 0 for j in range(num_labels)]
        yhat_raw_inds = [scors[hadm_id][ind2c[j]] if ind2c[j] in scors[hadm_id] else 0 for j in range(num_labels)]
        gold_inds = [1 if ind2c[j] in golds[hadm_id] else 0 for j in range(num_labels)]
        yhat[i] = yhat_inds
        yhat_raw[i] = yhat_raw_inds
        y[i] = gold_inds
    return yhat, yhat_raw, y, all_metrics(yhat, y, yhat_raw=yhat_raw, calc_auc=False)


def union_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)

def intersect_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)

def print_metrics(metrics, suffix=None, output_path=None):
    res = []
    for key in ['auc_macro', 'auc_micro', 'f1_macro', 'f1_micro', 
                'prec_at_5', 'prec_at_8', 'prec_at_15',
                'rec_at_5', 'rec_at_8', 'rec_at_15']:
        res.append(metrics.get(key, '-'))
    res = [format(r, '.4f') for r in res]
    if output_path is None:
        print('------')
        if suffix is not None:
            print(suffix)
        print('MACRO-auc, MICRO-auc, MACRO-f1, MICRO-f1, P@5, P@8, P@15, R@5, R@8, R@15')
        print(', '.join(res))
    else:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write('------\n')
            if suffix is not None:
                f.write(f'{suffix}\n')
            f.write('MACRO-auc, MICRO-auc, MACRO-f1, MICRO-f1, P@5, P@8, P@15, R@5, R@8, R@15\n')
            f.write(', '.join(res) + '\n')
            
    # main icd metric ooutput
    res = []
    if 'main_label_acc' in metrics:
        for key in ['main_label_acc', 'h@5', 'h@8', 'h@15', 'mrr']:
            res.append(metrics.get(key, '-'))
        res = [format(r, '.4f') for r in res]
        if output_path is None:
            print('------')
            if suffix is not None:
                print(suffix)
            print('main_label_acc, h@5, h@8, h@15, mrr')
            print(', '.join(res))
        else:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write('------\n')
                if suffix is not None:
                    f.write(f'{suffix}\n')
                f.write('main_label_acc, h@5, h@8, h@15, mrr\n')
                f.write(', '.join(res) + '\n') 

