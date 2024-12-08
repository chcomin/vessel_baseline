import inspect
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import get_dataset, collate_fn
from util import Logger, MultipleMetrics, seed_all, seed_worker, show_log
from metrics import confusion_metrics

class ModuleRunner:

    def __init__(
            self, 
            model, 
            dl_train, 
            dl_valid, 
            loss_func, 
            optim, 
            scheduler,
            perf_funcs,
            logger,
            device,
        ): 

        # Save all arguments as class attributes
        frame = inspect.currentframe()
        args, varargs, varkw, values = inspect.getargvalues(frame)
        for arg in args:
            setattr(self, arg, values[arg])
     
    def train_one_epoch(self, epoch):

        self.model.train()
        loss_log = 0.
        for batch_idx, (imgs, targets) in enumerate(self.dl_train):
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)
            self.optim.zero_grad()
            scores = self.model(imgs)
            loss = self.loss_func(scores, targets)
            loss.backward()
            self.optim.step()

            loss_log += loss.detach()*imgs.shape[0]

        loss_log /= len(self.dl_train.dataset)
        self.logger.log(epoch, 'train/loss', loss_log.item())
        self.scheduler.step()
    
    @torch.no_grad()
    def validate_one_epoch(self, epoch):

        dl_valid = self.dl_valid
     
        self.model.eval()
        loss_log = 0.
        perf_log = {}
        for batch_idx, (imgs, targets) in enumerate(dl_valid):
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)
            scores = self.model(imgs)
            loss = self.loss_func(scores, targets)

            loss_log += loss*imgs.shape[0]
            for perf_func in self.perf_funcs:
                results = perf_func(scores, targets)
                for name, value in results:
                    weighted_value = value*imgs.shape[0]
                    if name not in perf_log:
                        perf_log[name] = weighted_value
                    else:
                        perf_log[name] += weighted_value

        loss_log /= len(dl_valid.dataset)
        self.logger.log(epoch, 'valid/loss', loss_log.item())
        for name, value in perf_log.items():
            value = value/len(dl_valid.dataset)
            self.logger.log(epoch, f'valid/{name}', value.item())

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pass
    
    def predict(self, batch):
        
        model = self.model
        batch_device = batch.device

        model.eval()
        output = model(batch.to(model.device)).to(batch_device)

        return output

def train(
        model, 
        bs_train, 
        bs_valid, 
        num_epochs, 
        lr, 
        weight_decay=0., 
        resize_size=224, 
        seed=0,
        num_workers=0,
    ):

    seed_all(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds_train, ds_valid, class_weights = get_dataset(
        "/home/chcomin/Dropbox/ufscar/Visao Computacional/01-2024/Aulas/data/oxford_pets"
        #"E:/Dropbox/ufscar/Visao Computacional/01-2024/Aulas/data/oxford_pets"
        , resize_size=resize_size
        )
    #ds_train.indices = ds_train.indices[:5*32]
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
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optim, num_epochs)

    perf_funcs = [
        MultipleMetrics(['acc', 'iou', 'prec', 'rec'], confusion_metrics)
        ]
    logger = Logger()

    runner = ModuleRunner(
        model, 
        dl_train, 
        dl_valid, 
        loss_func, 
        optim, 
        scheduler,
        perf_funcs,
        logger,
        device
    )

    best_loss = torch.inf
    for epoch in range(0, num_epochs):
        runner.train_one_epoch(epoch)
        runner.validate_one_epoch(epoch)

        show_log(logger)

        checkpoint = {
            'params':{'bs_train':bs_train,'bs_valid':bs_valid,'lr':lr,
                      'weight_decay':weight_decay},
            'model':model.state_dict(),
            'optim':optim.state_dict(),
            'sched':scheduler.state_dict(),
            'logger':logger
        }

        torch.save(checkpoint, 'checkpoint.pt')
        #if loss_valid<best_loss:
        #    torch.save(checkpoint, 'best_model.pt')
        #    best_loss = loss_valid

    model.to('cpu')

    return ds_train, ds_valid, logger

