import torch
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

def dice_score(actual, predicted):
    actual = np.asarray(actual).astype(bool)
    predicted = np.asarray(predicted).astype(bool)
    im_sum = actual.sum() + predicted.sum()
    if im_sum == 0:
        return 1
    intersection = np.logical_and(actual, predicted)
    return 2. * intersection.sum() / im_sum

def evaluate(logits, labels, ignore_index = -100):

    all_probs_0 = []
    all_targets = []

    for logit, label in zip(logits, labels):
        # logits[i] is n_classes x h x w
        prob = torch.sigmoid(logit).detach().cpu().numpy()  # prob is n_classes x h x w
        target = label.cpu().numpy()

        all_probs_0.extend(prob.ravel())
        all_targets.append(target.ravel())

    all_probs_np = np.hstack(all_probs_0)
    all_targets_np = np.hstack(all_targets)

    all_probs_np = all_probs_np[all_targets_np != ignore_index]
    all_targets_np = all_targets_np[all_targets_np!=ignore_index]

    all_preds_np = all_probs_np > 0.5
    return roc_auc_score(all_targets_np, all_probs_np), f1_score(all_targets_np, all_preds_np)
