import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import display
import torch

class BatchMetric:
    """"
    Class for storing batch metrics during an epoch. The class allows storing a 
    single value for each batch or an iterable of values containing a metric 
    calculated for each item of the batches.

    Example 1:
    metric = BatchMetric('train/loss')
    ...
    loss = loss_func(scores, targets)
    metric.add(loss, 32)  # Average loss calculated over 32 items
    ...
    average = metric.reduce() # Average loss over all batches in epoch

    Example 2:
    metric = BatchMetric('train/loss', per_item_metrics=True)
    ...
    loss = loss_func(scores, targets)  # loss created with reduction='none'
    metric.add(loss)  // Loss value for each item in the batch
    ...
    average = metric.reduce() # Average loss over all batches in epoch
    """

    def __init__(self, name, per_item_metrics=False):
        self.name = name
        self.per_item_metrics = per_item_metrics
        self.values = []

    def add(self, value, n_items=1):

        per_item_metrics = self.per_item_metrics
        if per_item_metrics and not self._is_iterable(value):
            raise ValueError('Since per_item_metrics is True, value must be iterable.')
        if per_item_metrics and n_items>1:
            raise ValueError('n_items only allowed if per_item_metrics is False.')
        if not per_item_metrics and self._is_iterable(value):
            raise ValueError('Since per_item_metrics is False, value must be scalar.')
        
        if per_item_metrics:
            # value is iterable and has a length
            data = (value, None)
        else:
            # value is scalar and may represent an average over n_items
            data = (value, n_items)

        self.values.append(data)

    def reduce(self, reduction='mean'):

        agg_metric = 0
        n_elements = 0
        for value, n_items in self.values:
            if self.per_item_metrics:
                # Sum values in batch and count number of elements
                batch_sum = sum(value)
                n_elements += len(value)
            else:
                # Multiply value by n_items to undo the average
                batch_sum = value*n_items
                n_elements += n_items

            # Check if batch_sum has the method item() to convert the value
            # to a Python scalar and send it to the cpu. If the method is
            # absent, assume the value is already a Python scalar.
            try:
                batch_sum = batch_sum.item()
            except AttributeError:
                pass

            agg_metric += batch_sum

        if reduction=='mean':
            agg_metric /= n_elements

        return agg_metric

    def _is_iterable(self, obj):
        # obj can be a python, tensor or numpy scalar as well as
        # a list, tuple, tensor or numpy array. We need to handle all cases.
        try:
            iter(obj)
            return True
        except TypeError:
            return False

class Logger:

    def __init__(self): 
        self.epoch_data = {}
        self.current_epoch = 0
        #self.names = metric_names
            
    def log(self, epoch, name, value):

        if epoch!=self.current_epoch and epoch!=self.current_epoch+1:
            raise ValueError(f'Current epoch is {self.current_epoch} but {epoch} received')
        #if name not in self.names:
        #    raise ValueError(f'Metric {name} not in list of metric names.')

        epoch_data = self.epoch_data
        if epoch not in epoch_data:
            epoch_data[epoch] = {}
            self.current_epoch = epoch

        epoch_data[epoch][name] = value

    def get_data(self):

        df = pd.DataFrame(self.epoch_data).T
        df.insert(0, 'epoch', df.index)

        return df

class SingleMetric:

    def __init__(self, metric_name, func):
        self.metric_name = metric_name
        self.func = func

    def __call__(self, *args):
        return (self.metric_name, self.func(*args))

class MultipleMetrics:

    def __init__(self, metric_names, func):
        self.metric_names = metric_names
        self.func = func

    def __call__(self, *args):
        results = self.func(*args)
        return ((name,result) for name, result in zip(self.metric_names, results))

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

    df = logger.get_data()
    epochs = df['epoch']
    train_loss = df['train/loss']
    valid_loss = df['valid/loss']
    acc_names = df.columns[3:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,3))
    ax1.plot(epochs, train_loss, '-o', ms=2, label='Train loss')
    ax1.plot(epochs, valid_loss, '-o', ms=2, label='Valid loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_ylim((0,1.))
    ax1.legend()

    for name in acc_names:
        ax2.plot(epochs, df[name], '-o', ms=2, label=name)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(name)
        ax2.set_ylim((0,1.))
        ax2.legend()
    fig.tight_layout()

    display.clear_output(wait=True)
    plt.show()