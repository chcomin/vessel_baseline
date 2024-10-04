import torch
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
from scipy.stats import rankdata

def dice_score(actual, predicted):
    actual = np.asarray(actual).astype(bool)
    predicted = np.asarray(predicted).astype(bool)
    im_sum = actual.sum() + predicted.sum()
    if im_sum == 0:
        return 1
    intersection = np.logical_and(actual, predicted)
    return 2. * intersection.sum() / im_sum

def accuracy_score(actual, predicted):
    actual = np.asarray(actual).astype(bool)
    predicted = np.asarray(predicted).astype(bool)
    num_els = actual.size
    intersection = np.logical_and(actual, predicted)
    return float(intersection.sum()) / num_els

def fast_auc(actual, predicted):
    r = rankdata(predicted)
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    return (np.sum(r[actual==1]) - n_pos*(n_pos+1)/2) / n_pos / n_neg

def evaluate(logits, labels, n_classes, ignore_index = -100, fast=False):

    all_probs_0 = []
    all_targets = []


    if n_classes == 4:
        all_probs_1 = []
        all_probs_2 = []
        all_probs_3 = []

    act = torch.sigmoid if n_classes==1 else torch.nn.Softmax(dim=0)

    for logit, label in zip(logits, labels):
        # prediction = act(logits[i]).detach().cpu().numpy()[-1]  # this takes last channel in multi-class, ok for 2-class
        # logits[i] is n_classes x h x w
        prob = act(logit).detach().cpu().numpy()  # prob is n_classes x h x w
        target = label.cpu().numpy()

        if n_classes==1:
            all_probs_0.extend(prob.ravel())
        else:
            all_probs_0.extend(prob[0].ravel())
            all_probs_1.extend(prob[1].ravel())
            all_probs_2.extend(prob[2].ravel())
            all_probs_3.extend(prob[3].ravel())

        all_targets.append(target.ravel())

    if n_classes == 1:
        all_probs_np = np.hstack(all_probs_0)
    else: all_probs_np = np.stack([all_probs_0, all_probs_1, all_probs_2, all_probs_3], axis=1)

    all_targets_np = np.hstack(all_targets)

    all_probs_np = all_probs_np[all_targets_np != ignore_index]
    all_targets_np = all_targets_np[all_targets_np!=ignore_index]

    if n_classes == 4:
        all_preds_np = np.argmax(all_probs_np, axis=1)
        return roc_auc_score(all_targets_np, all_probs_np, multi_class='ovo',average='weighted'), f1_score(all_targets_np, all_preds_np,average='weighted')
    else:
        all_preds_np = all_probs_np > 0.5
        if fast:
            return fast_auc(all_targets_np>0.5, all_probs_np), f1_score(all_targets_np>0.5, all_preds_np)
        else:
            return roc_auc_score(all_targets_np, all_probs_np), f1_score(all_targets_np, all_preds_np)
