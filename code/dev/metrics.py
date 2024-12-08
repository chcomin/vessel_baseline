import torch

@torch.no_grad()
def confusion_metrics(scores, targets):
    """Calculate accuracy, precision, recall, IoU and Dice scores for a batch
    of data.

    Parameters
    ----------
    scores
        Output from a network. Dimension 1 is treated as the class dimension.
    targets
        Labels

    Returns
    -------
        A tuple (acc, iou, prec, rec, dice) containing the accuracy, IoU, 
        precision, recall, and Dice scores.
    """

    pred = scores.argmax(dim=1).reshape(-1)
    targets = targets.reshape(-1)

    pred = pred[targets!=2]
    targets = targets[targets!=2]

    pred = pred>0
    targets = targets>0
    tp = (targets & pred).sum()
    tn = (~targets & ~pred).sum()
    fp = (~targets & pred).sum()
    fn = (targets & ~pred).sum()

    acc = (tp+tn)/(tp+tn+fp+fn)
    iou = tp/(tp+fp+fn)
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    dice = 2*tp/(2*tp+fn)

    return acc, iou, prec, rec, dice
