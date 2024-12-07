import random
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import get_dataset, collate_fn

def seed_all(seed, deterministic=True):
    """
    Seed all random number generators for reproducibility. If deterministic is
    True, set cuDNN to deterministic mode.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    """
    Set Python and numpy seeds for dataloader workers. Each worker receives a 
    different seed in initial_seed().
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def show_log(logger):
    """
    Plota métricas em um notebook.
    """

    epochs, losses_train, losses_valid, accs = zip(*logger)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,3))
    ax1.plot(epochs, losses_train, '-o', ms=2, label='Train loss')
    ax1.plot(epochs, losses_valid, '-o', ms=2, label='Valid loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_ylim((0,1.))
    ax1.legend()
    ax2.plot(epochs, accs, '-o', ms=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim((0,1.))
    fig.tight_layout()

    display.clear_output(wait=True)
    plt.show()

@torch.no_grad()
def iou(scores, targets):
    """
    Função que calcula a Intersecção sobre a União entre o resultado
    da rede e o rótulo conhecido.
    """

    pred = scores.argmax(dim=1).reshape(-1)
    targets = targets.reshape(-1)

    pred = pred[targets!=2]
    targets = targets[targets!=2]

    tp = ((targets==1) & (pred==1)).sum()
    tn = ((targets==0) & (pred==0)).sum()
    fp = ((targets==0) & (pred==1)).sum()
    fn = ((targets==1) & (pred==0)).sum()

    acc = (tp+tn)/(tp+tn+fp+fn)
    iou = tp/(tp+fp+fn)
    prec = tp/(tp+fp)
    rev = tp/(tp+fn)

    return iou

def train_step(model, dl_train, optim, loss_func, scheduler, device):
    """
    Executa uma época de treinamento.
    """

    model.train()
    loss_log = 0.
    idx = 0
    for imgs, targets in dl_train:
        idx += 1
        imgs = imgs.to(device)
        targets = targets.to(device)
        optim.zero_grad()
        scores = model(imgs)
        loss = loss_func(scores, targets)
        loss.backward()
        optim.step()

        loss_log += loss.detach()*imgs.shape[0]

    scheduler.step()
    loss_log /= len(dl_train.dataset)

    return loss_log.item()

@torch.no_grad()
def valid_step(model, dl_valid, loss_func, perf_func, device):

    model.eval()
    loss_log = 0.
    perf_log = 0.
    for imgs, targets in dl_valid:
        imgs = imgs.to(device)
        targets = targets.to(device)
        scores = model(imgs)
        loss = loss_func(scores, targets)
        perf = perf_func(scores, targets)

        loss_log += loss*imgs.shape[0]
        perf_log += perf*imgs.shape[0]

    loss_log /= len(dl_valid.dataset)
    perf_log /= len(dl_valid.dataset)

    return loss_log.item(), perf_log.item()

def train(
        model, 
        bs_train, 
        bs_valid, 
        num_epochs, 
        lr, 
        weight_decay=0., 
        resize_size=224, 
        seed=0,
        num_workers=5,
    ):

    seed_all(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds_train, ds_valid, class_weights = get_dataset(
        "/home/chcomin/Dropbox/ufscar/Visao Computacional/01-2024/Aulas/data/oxford_pets"
        #"E:/Dropbox/ufscar/Visao Computacional/01-2024/Aulas/data/oxford_pets"
        , resize_size=resize_size
        )
    #ds_train.indices = ds_train.indices[:5*256]
    model.to(device)

    dl_train = DataLoader(
        ds_train, 
        batch_size=bs_train, 
        shuffle=True, 
        num_workers=num_workers,
        persistent_workers=num_workers>0,
        worker_init_fn=seed_worker,
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=bs_valid, 
        shuffle=False, 
        collate_fn=collate_fn,
       num_workers=num_workers, 
       persistent_workers=num_workers>0
    )

    loss_func = nn.CrossEntropyLoss(torch.tensor(class_weights, device=device), ignore_index=2)
    optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                            momentum=0.9)
    sched = torch.optim.lr_scheduler.PolynomialLR(optim, num_epochs)
    logger = []
    best_loss = torch.inf
    for epoch in range(0, num_epochs):
        loss_train = train_step(model, dl_train, optim, loss_func, sched, device)
        loss_valid, perf = valid_step(model, dl_valid, loss_func, iou, device)
        logger.append((epoch, loss_train, loss_valid, perf))

        show_log(logger)

        checkpoint = {
            'params':{'bs_train':bs_train,'bs_valid':bs_valid,'lr':lr,
                      'weight_decay':weight_decay},
            'model':model.state_dict(),
            'optim':optim.state_dict(),
            'sched':sched.state_dict(),
            'logger':logger
        }

        torch.save(checkpoint, 'checkpoint.pt')
        if loss_valid<best_loss:
            torch.save(checkpoint, 'best_model.pt')
            best_loss = loss_valid

    model.to('cpu')

    return ds_train, ds_valid, logger