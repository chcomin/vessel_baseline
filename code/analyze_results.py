import argparse
import sys
import os
import os.path as osp
import pandas as pd
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.metrics import matthews_corrcoef
from skimage import img_as_float
from utils.evaluation import dice_score

# future-self: dice and f1 are the same thing, but if you use f1_score from sklearn it will be much slower, the reason
# being that dice here expects bools and it won't work in multi-class scenarios. Same goes for accuracy_score.
# (see https://brenocon.com/blog/2012/04/f-scores-dice-and-jaccard-set-similarity/)

def get_labels_preds(path_to_preds, csv_path):
    df = pd.read_csv(csv_path)
    im_paths, mask_paths, gt_paths = df.im_paths, df.mask_paths, df.gt_paths
    root = osp.dirname(csv_path)
    all_preds = []
    all_gts = []
    for im_path, gt_path, mask_path in zip(im_paths, gt_paths, mask_paths):
        im_path = im_path.rsplit('/', 1)[-1]
        pred_path = osp.join(path_to_preds, im_path[:-4] + '.png')

        gt = np.array(Image.open(osp.join(root,gt_path))).astype(bool)
        mask = np.array(Image.open(osp.join(root,mask_path)).convert('L')).astype(bool)
        try:
            pred = img_as_float(np.array(Image.open(pred_path)))
        except FileNotFoundError:
            sys.exit(f'---- no predictions found at {path_to_preds} (maybe run first generate_results.py?) ---- ')
        gt_flat = gt.ravel()
        mask_flat = mask.ravel()
        pred_flat = pred.ravel()
        # do not consider pixels out of the FOV
        noFOV_gt = gt_flat[mask_flat == True]
        noFOV_pred = pred_flat[mask_flat == True]

        # accumulate gt pixels and prediction pixels
        all_preds.append(noFOV_pred)
        all_gts.append(noFOV_gt)

    return np.hstack(all_preds), np.hstack(all_gts)

def cutoff_youden(fpr, tpr, thresholds):
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

def cutoff_dice(preds, gts):
    dice_scores = []
    thresholds = np.linspace(0, 1, 256)
    for i in tqdm(range(len(thresholds))):
        thresh = thresholds[i]
        hard_preds = preds>thresh
        dice_scores.append(dice_score(gts, hard_preds))
    dices = np.array(dice_scores)
    optimal_threshold = thresholds[dices.argmax()]
    return optimal_threshold

def cutoff_accuracy(preds, gts):
    accuracy_scores = []
    thresholds = np.linspace(0, 1, 256)
    for i in tqdm(range(len(thresholds))):
        thresh = thresholds[i]
        hard_preds = preds > thresh
        accuracy_scores.append(accuracy_score(gts.astype(np.bool), hard_preds.astype(np.bool)))
    accuracies = np.array(accuracy_scores)
    optimal_threshold = thresholds[accuracies.argmax()]
    return optimal_threshold

def compute_performance(preds, gts, save_path=None, opt_threshold=None, cut_off='dice', mode='train'):

    fpr, tpr, thresholds = roc_curve(gts, preds)
    global_auc = auc(fpr, tpr)

    if save_path is not None:
        fig = plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, label='ROC curve')
        ll = f'AUC = {global_auc:4f}'
        plt.legend([ll], loc='lower right')
        fig.tight_layout()
        if opt_threshold is None:
            if mode=='train':
                plt.savefig(osp.join(save_path,'ROC_train.png'))
            elif mode=='val':
                plt.savefig(osp.join(save_path, 'ROC_val.png'))
        else:
            plt.savefig(osp.join(save_path, 'ROC_test.png'))

    if opt_threshold is None:
        if cut_off == 'acc':
            # this would be to get accuracy-maximizing threshold
            opt_threshold = cutoff_accuracy(preds, gts)
        elif cut_off == 'dice':
            # this would be to get dice-maximizing threshold
            opt_threshold = cutoff_dice(preds, gts)
        else:
            opt_threshold = cutoff_youden(fpr, tpr, thresholds)

    #opt_threshold = 0.5
    bin_preds = preds > opt_threshold

    acc = accuracy_score(gts, bin_preds)

    dice = dice_score(gts, bin_preds)

    mcc = matthews_corrcoef(gts.astype(int), bin_preds.astype(int))

    tn, fp, fn, tp = confusion_matrix(gts, preds > opt_threshold).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    return global_auc, acc, dice, mcc, specificity, sensitivity, opt_threshold

def main(args):

    dataset = args.dataset
    path_preds = args.path_preds
    split = args.split
    cut_off = args.cut_off

    print(f'* Analyzing performance in {dataset} training set -- Obtaining optimal threshold maximizing {cut_off}')
    print(f'* Reading predictions from {path_preds}')
    save_path = osp.join(path_preds, 'perf')

    perf_csv_path = osp.join(save_path, 'training_performance.csv')
    csv_path = osp.join('../data', dataset, f'train{split}.csv')
    preds, gts = get_labels_preds(path_preds, csv_path = csv_path)
    os.makedirs(save_path, exist_ok=True)
    metrics = compute_performance(preds, gts, save_path=save_path, opt_threshold=None, cut_off=cut_off, mode='train')
    global_auc_tr, acc_tr, dice_tr, mcc_tr, spec_tr, sens_tr, opt_thresh_tr = metrics

    perf_df_train = pd.DataFrame({'auc': global_auc_tr,
                                'acc': acc_tr,
                                'dice/F1': dice_tr,
                                'MCC': mcc_tr,
                                'spec': spec_tr,
                                'sens': sens_tr,
                                'opt_t': opt_thresh_tr}, index=[0])
    perf_df_train.to_csv(perf_csv_path, index=False)

    print(f'* Analyzing performance in {dataset} validation set')
    perf_csv_path = osp.join(save_path, 'validation_performance.csv')
    csv_path = osp.join('../data', dataset, f'val{split}.csv')
    preds, gts = get_labels_preds(path_preds, csv_path = csv_path)
    metrics = compute_performance(preds, gts, save_path=save_path, opt_threshold=opt_thresh_tr, cut_off=cut_off, mode='train')
    global_auc_vl, acc_vl, dice_vl, mcc_vl, spec_vl, sens_vl, _ = metrics
    perf_df_train = pd.DataFrame({'auc': global_auc_vl,
                                  'acc': acc_vl,
                                  'dice/F1': dice_vl,
                                  'MCC': mcc_vl,
                                  'spec': spec_vl,
                                  'sens': sens_vl}, index=[0])
    perf_df_train.to_csv(perf_csv_path, index=False)

    print(f'*Analyzing performance in {dataset} test set')
    print(f'* Reading predictions from {path_preds}')
    save_path = osp.join(path_preds, 'perf')
    os.makedirs(save_path, exist_ok=True)
    perf_csv_path = osp.join(save_path, 'test_performance.csv')

    csv_name = f'test{split}.csv'
    print('-- Testing')

    path_test_csv = osp.join('../data', dataset, csv_name)

    preds, gts = get_labels_preds(path_preds, csv_path = path_test_csv)
    global_auc_test, acc_test, dice_test, mcc_test, spec_test, sens_test, _ = \
        compute_performance(preds, gts, save_path=save_path, opt_threshold=opt_thresh_tr)
    perf_df_test = pd.DataFrame({'auc': global_auc_test,
                                 'acc': acc_test,
                                 'dice/F1': dice_test,
                                 'MCC': mcc_test,
                                 'spec': spec_test,
                                 'sens': sens_test}, index=[0])
    perf_df_test.to_csv(perf_csv_path, index=False)
    print('* Done')
    print(f'AUC in Train/Val/Test set is {global_auc_tr:.4f}/{global_auc_vl:.4f}/{global_auc_test:.4f}')
    print(f'Accuracy in Train/Val/Test set is {acc_tr:.4f}/{acc_vl:.4f}/{acc_test:.4f}')
    print(f'Dice/F1 score in Train/Val/Test set is {dice_tr:.4f}/{dice_vl:.4f}/{dice_test:.4f}')
    print(f'MCC score in Train/Val/Test set is {mcc_tr:.4f}/{mcc_vl:.4f}/{mcc_test:.4f}')
    print(f'Specificity in Train/Val/Test set is {spec_tr:.4f}/{spec_vl:.4f}/{spec_test:.4f}')
    print(f'Sensitivity in Train/Val/Test set is {sens_tr:.4f}/{sens_vl:.4f}/{sens_test:.4f}')
    print('ROC curve plots saved to ', save_path)
    print('Perf csv saved at ', perf_csv_path)

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DRIVE', help='which dataset to test')
    parser.add_argument('--path_preds', type=str, default=None, help='path to training predictions')
    parser.add_argument('--split', type=str, default='', help='name of split to use')
    parser.add_argument('--cut_off', type=str, default='dice', help='threshold maximizing x, x=dice/acc/youden')

    return parser.parse_args()

if __name__ == '__main__':

    _args = get_args()
    main(_args)
