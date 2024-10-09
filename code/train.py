import sys
import json
import os
import os.path as osp
import random
import argparse
from datetime import datetime
import operator
from tqdm import tqdm
import numpy as np
import torch

from models.get_model import get_arch
from utils.get_loaders import get_train_val_loaders
from utils.evaluation import evaluate
from utils.model_saving_loading import save_model, str2bool

def compare_op(metric):
    """
    This should return an operator that given a, b returns True if a is better than b
    Also, let us return which is an appropriately terrible initial value for such metric
    """

    if metric == 'auc':
        op, init = operator.gt, 0
    elif metric == 'tr_auc':
        op, init = operator.gt, 0
    elif metric == 'dice':
        op, init = operator.gt, 0
    elif metric == 'loss':
        op, init = operator.lt, np.inf
    else:
        raise NotImplementedError

    return op, init

def set_seeds(seed_value, use_cuda):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        # torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False

def reduce_lr(optimizer, epoch, factor=0.1, verbose=True):
    for i, param_group in enumerate(optimizer.param_groups):
        old_lr = float(param_group['lr'])
        new_lr = old_lr * factor
        param_group['lr'] = new_lr
        if verbose:
            print(f'Epoch {epoch:5d}: reducing learning rate of group {i} to {new_lr:.4e}.')

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def run_one_epoch(
        loader,
        model,
        criterion,
        optimizer=None,
        scheduler=None,
        assess=False
        ):

    device='cuda' if next(model.parameters()).is_cuda else 'cpu'
    train = optimizer is not None  # if we are in training mode there will be an optimizer and train=True here

    if train:
        model.train()
    else:
        model.eval()

    if assess:
        logits_all, labels_all = [], []
    n_elems, running_loss, tr_lr = 0, 0, 0


    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        if model.n_classes == 1:
            loss = criterion(logits, labels.unsqueeze(dim=1).float())  # BCEWithLogitsLoss()/DiceLoss()
        else:
            loss = criterion(logits, labels)  # CrossEntropyLoss()

        if train:  # only in training mode
            optimizer.zero_grad()
            loss.backward()
            tr_lr = get_lr(optimizer)
            optimizer.step()
            scheduler.step()
        if assess:
            logits_all.extend(logits)
            labels_all.extend(labels)

        # Compute running loss
        running_loss += loss.item() * inputs.size(0)
        n_elems += inputs.size(0)
        run_loss = running_loss / n_elems

    if assess:
        return logits_all, labels_all, run_loss, tr_lr
    return None, None, run_loss, tr_lr

def train_one_cycle(train_loader, model, criterion, optimizer=None, scheduler=None, cycle=0):

    model.train()
    cycle_len = scheduler.cycle_lens[cycle]

    with tqdm(range(cycle_len)) as t:
        for epoch in t:
            if epoch == cycle_len-1:
                assess=True # only get logits/labels on last cycle
            else: assess = False
            tr_logits, tr_labels, tr_loss, tr_lr = run_one_epoch(train_loader, model, criterion, optimizer=optimizer,
                                                          scheduler=scheduler, assess=assess)
            t.set_postfix(tr_loss_lr=f"{float(tr_loss):.4f}/{tr_lr:.6f}")

    return tr_logits, tr_labels, tr_loss

def train_model(model, optimizer, criterion, train_loader, val_loader, epochs, evaluate_every, scheduler, metric, exp_path):

    best_auc, best_dice, best_epoch = 0, 0, 0
    is_better, best_monitoring_metric = compare_op(metric)

    with tqdm(range(1, epochs+1), initial=1) as t:
        for epoch in t:
            model.train()
            if epoch%evaluate_every==0:
                assess=True # only get logits/labels when evaluating
            else:
                assess = False
            tr_logits, tr_labels, tr_loss, tr_lr = run_one_epoch(train_loader, model, criterion, optimizer=optimizer,
                                                        scheduler=scheduler, assess=assess)
            
            t.set_postfix(epoch=f"{epoch:d}/{epochs:d}", tr_loss_lr=f"{float(tr_loss):.4f}/{tr_lr:.6f}")

            if assess:
                print(25 * '-' + '  Evaluating ' + 25 * '-')
                tr_auc, tr_dice = evaluate(tr_logits, tr_labels)

                with torch.no_grad():
                    assess=True
                    vl_logits, vl_labels, vl_loss, _ = run_one_epoch(val_loader, model, criterion, assess=assess)
                    vl_auc, vl_dice = evaluate(vl_logits, vl_labels)
                    del vl_logits, vl_labels
                msg = f'Train/Val Loss: {tr_loss:.4f}/{vl_loss:.4f}  -- Train/Val AUC: {tr_auc:.4f}/{vl_auc:.4f}  -- Train/Val DICE: {tr_dice:.4f}/{vl_dice:.4f} -- LR={get_lr(optimizer):.6f}'
                print(msg.rstrip('0'))

                # check if performance was better than anyone before and checkpoint if so
                if metric == 'auc':
                    monitoring_metric = vl_auc
                elif metric == 'tr_auc':
                    monitoring_metric = tr_auc
                elif metric == 'loss':
                    monitoring_metric = vl_loss
                elif metric == 'dice':
                    monitoring_metric = vl_dice
                if is_better(monitoring_metric, best_monitoring_metric):
                    print(f'Best {metric} attained. {100*best_monitoring_metric:.2f} --> {100*monitoring_metric:.2f}')
                    best_auc, best_dice, best_epoch = vl_auc, vl_dice, epoch
                    best_monitoring_metric = monitoring_metric
                    if exp_path is not None:
                        print(25 * '-', ' Checkpointing ', 25 * '-')
                        save_model(exp_path, model, optimizer)

    del model
    torch.cuda.empty_cache()
    
    return best_auc, best_dice, best_epoch

def main(args):

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("cuda is not currently available!")
        device = torch.device("cuda")
    else:  #cpu
        device = torch.device(args.device)

    # reproducibility
    seed_value = args.seed
    set_seeds(seed_value, args.device.startswith("cuda"))

    # gather parser parameters
    model_name = args.model_name
    max_lr, bs = args.max_lr, args.batch_size
    epochs, evaluate_every = args.epochs, args.evaluate_every
    metric = args.metric
    in_c = args.in_c
    use_green = args.use_green

    if in_c==3 and use_green==True:
        raise ValueError('Parameter use_green was changed to True, but in_c=3.')
    if in_c==3:
        channels = 'all'
    elif in_c==1:
        if use_green:
            channels = 'green'
        else:
            channels = 'gray'

    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size)==1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size)==2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')

    do_not_save = str2bool(args.do_not_save)
    if do_not_save is False:
        save_path = args.save_path
        if save_path == 'date_time':
            save_path = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        experiment_path=osp.join('../experiments', save_path)
        args.experiment_path = experiment_path
        os.makedirs(experiment_path, exist_ok=True)

        config_file_path = osp.join(experiment_path,'config.cfg')
        with open(config_file_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    else:
        experiment_path = None

    csv_train = args.csv_train
    csv_val = csv_train.replace('train', 'val')

    n_classes = 1
    label_values = [0, 255]


    print(f"* Creating Dataloaders, batch size = {bs}, workers = {args.num_workers}")
    train_loader, val_loader = get_train_val_loaders(csv_path_train=csv_train, csv_path_val=csv_val, batch_size=bs, 
                                                     tg_size=tg_size, label_values=label_values, channels=channels,
                                                     num_workers=args.num_workers)

    print(f'* Instantiating a {model_name} model')
    model = get_arch(model_name, in_c=in_c, n_classes=n_classes)
    model = model.to(device)

    num_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {num_p:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)

    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=epochs*len(train_loader), power=0.9)
    setattr(optimizer, 'max_lr', max_lr)  # store it inside the optimizer for accessing to it later

    criterion = torch.nn.BCEWithLogitsLoss() if model.n_classes == 1 else torch.nn.CrossEntropyLoss()

    print('* Instantiating loss function', str(criterion))
    print('* Starting to train\n','-' * 10)

    m1, m2, m3 = train_model(model, optimizer, criterion, train_loader, val_loader, epochs, evaluate_every, scheduler, metric, experiment_path)

    print(f"val_auc: {m1}")
    print(f"val_dice: {m2}")
    print(f"best_epoch: {m3}")
    if do_not_save is False:

        with open(osp.join(experiment_path, 'val_metrics.txt'), 'w') as f:
            print(f'Best AUC = {100*m1:.2f}\nBest DICE = {100*m2:.2f}\nBest epoch = {m3}', file=f)

def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--csv_train', type=str, default='data/DRIVE/train.csv', help='path to training data csv')
    parser.add_argument('--model_name', type=str, default='unet', help='architecture')
    parser.add_argument('--batch_size', type=int, default=4, help='batch Size')
    parser.add_argument('--max_lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--evaluate_every', type=int, default=50, help='number of epochs between model evaluations')
    parser.add_argument('--metric', type=str, default='auc', help='which metric to use for monitoring progress (tr_auc/auc/loss/dice)')
    parser.add_argument('--im_size', help='delimited list input, could be 600,400', type=str, default='512')
    parser.add_argument('--in_c', type=int, default=3, help='channels in input images')
    parser.add_argument('--use_green', type=int, default=0, help='if 0 and in_c=1, converts to gray. Use green channel otherwise')
    parser.add_argument('--do_not_save', type=str2bool, nargs='?', const=True, default=False, help='avoid saving anything')
    parser.add_argument('--save_path', type=str, default='date_time', help='path to save model (defaults to date/time')
    parser.add_argument('--num_workers', type=int, default=0, help='number of parallel (multiprocessing) workers to launch for data loading tasks (handled by pytorch) [default: %(default)s]')
    parser.add_argument('--device', type=str, default='cuda:0', help='where to run the training code (e.g. "cpu" or "cuda:0") [default: %(default)s]')
    parser.add_argument('--seed', type=int, default=0, help='seed')

    return parser

if __name__ == '__main__':

    _args = get_parser().parse_args()
    main(_args)
