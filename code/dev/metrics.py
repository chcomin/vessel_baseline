import torch

@torch.no_grad()
def confusion_metrics(scores, targets):
    """
    Função que calcula a Intersecção sobre a União entre o resultado
    da rede e o rótulo conhecido.
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

    return acc, iou, prec, rec
