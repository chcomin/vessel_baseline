import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import display
import torch

class Logger:
    """
    Class for logging metrics during training and validation.
    """

    def __init__(self): 
        self.epoch_data = {}
        self.current_epoch = 0
        #self.names = metric_names
            
    def log(self, epoch, name, value):
        """Log a metric value for a given epoch.

        Parameters
        ----------
        epoch : int
            Epoch number
        name : str
            Name of the metric
        value : float | int
            Value to be logged
        """

        if epoch!=self.current_epoch and epoch!=self.current_epoch+1:
            raise ValueError(f'Current epoch is {self.current_epoch} but {epoch} received')

        epoch_data = self.epoch_data
        if epoch not in epoch_data:
            epoch_data[epoch] = {}
            self.current_epoch = epoch

        epoch_data[epoch][name] = value

    def get_data(self):
        """Returns a pandas dataframe with the logged data.

        Returns
        -------
        pd.DataFrame
            The dataframe
        """

        df = pd.DataFrame(self.epoch_data).T
        df.insert(0, 'epoch', df.index)

        return df

class SingleMetric:
    """
    Class for storing a function representing a performance metric.
    """

    def __init__(self, metric_name, func):
        """
        Create a SingleMetric object from a performance function.

        Parameters
        ----------
        metric_name : Name of the metric
        func : Function that calculates the metric
        """
        self.metric_name = metric_name
        self.func = func

    def __call__(self, *args):
        return (self.metric_name, self.func(*args))

class MultipleMetrics:
    """
    Class for storing a function that calculates many performance metrics in one call.
    """

    def __init__(self, metric_names, func):
        """
        Create a MultipleMetrics object from a performance function.

        Parameters
        ----------
        metric_name : Name of the metric
        func : Function that calculates the metric
        """
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
    Plot the logged data from a Logger object in a Jupyter notebook.
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