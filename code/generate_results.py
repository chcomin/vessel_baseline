import json
import sys
import os
import os.path as osp
import argparse
import warnings
import time
from tqdm import tqdm
import numpy as np
from skimage.io import imsave
from skimage.util import img_as_ubyte
import skimage.transform
import torch
from models.get_model import get_arch
from utils.get_loaders import get_test_dataset
from utils.model_saving_loading import load_model

def flip_ud(tens):
    return torch.flip(tens, dims=[1])

def flip_lr(tens):
    return torch.flip(tens, dims=[2])

def flip_lrud(tens):
    return torch.flip(tens, dims=[1, 2])

def create_pred(model, tens, mask, coords_crop, original_sz, tta='no'):
    act = torch.sigmoid if model.n_classes == 1 else torch.nn.Softmax(dim=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        logits = model(tens.unsqueeze(dim=0).to(device)).squeeze(dim=0)
    pred = act(logits)

    if tta!='no':
        with torch.no_grad():
            logits_lr = model(tens.flip(-1).unsqueeze(dim=0).to(device)).squeeze(dim=0).flip(-1)
            logits_ud = model(tens.flip(-2).unsqueeze(dim=0).to(device)).squeeze(dim=0).flip(-2)
            logits_lrud = model(tens.flip(-1).flip(-2).unsqueeze(dim=0).to(device)).squeeze(dim=0).flip(-1).flip(-2)

        if tta == 'from_logits':
            mean_logits = torch.mean(torch.stack([logits, logits_lr, logits_ud, logits_lrud]), dim=0)
            pred = act(mean_logits)
        elif tta == 'from_preds':
            pred_lr = act(logits_lr)
            pred_ud = act(logits_ud)
            pred_lrud = act(logits_lrud)
            pred = torch.mean(torch.stack([pred, pred_lr, pred_ud, pred_lrud]), dim=0)
        else: raise NotImplementedError
    pred = pred.detach().cpu().numpy()[-1]  # this takes last channel in multi-class, ok for 2-class
    # Orders: 0: NN, 1: Bilinear(default), 2: Biquadratic, 3: Bicubic, 4: Biquartic, 5: Biquintic
    pred = skimage.transform.resize(pred, output_shape=original_sz, order=3)
    full_pred = np.zeros_like(mask, dtype=float)
    full_pred[coords_crop[0]:coords_crop[2], coords_crop[1]:coords_crop[3]] = pred
    full_pred[~mask.astype(bool)] = 0

    return full_pred

def save_pred(full_pred, save_results_path, im_name):
    os.makedirs(save_results_path, exist_ok=True)
    im_name = im_name.rsplit('/', 1)[-1]
    save_name = osp.join(save_results_path, im_name[:-4] + '.png')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # this casts preds to int, loses precision but meh
        imsave(save_name, img_as_ubyte(full_pred))

def main(args):

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("cuda is not currently available!")
        device = torch.device("cuda")
    else:  #cpu
        device = torch.device(args.device)

    dataset = args.dataset
    split = args.split
    tta = args.tta

    # parse config file if provided
    config_file = args.config_file
    if config_file is not None:
        if not osp.isfile(config_file):
            raise ValueError('non-existent config file')
        with open(args.config_file, 'r') as f:
            args.__dict__.update(json.load(f))
    experiment_path = args.experiment_path # these should exist in a config file
    model_name = args.model_name
    in_c = args.in_c
    use_green = args.use_green

    if in_c==3 and use_green==1:
        raise ValueError('Parameter use_green was changed to True, but in_c=3.')
    if in_c==3:
        channels = 'all'
    elif in_c==1:
        if use_green==1:
            channels = 'green'
        else:
            channels = 'gray'

    if experiment_path is None:
        raise ValueError('must specify path to experiment')

    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size)==1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size)==2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')

    data_path = osp.join('../data', dataset)

    csv_path = f'test_all{split}.csv'
    print('* Reading test data from ' + osp.join(data_path, csv_path))
    test_dataset = get_test_dataset(data_path, csv_path=csv_path, tg_size=tg_size, channels=channels)
    print(f'* Instantiating model  = {str(model_name)}')
    model = get_arch(model_name, in_c=in_c).to(device)

    print('* Loading trained weights from ' + experiment_path)
    try:
        model, stats, _ = load_model(model, experiment_path, device)
    except RuntimeError:
        sys.exit('---- bad config specification (check layers, n_classes, etc.) ---- ')
    model.eval()

    save_results_path = osp.join(args.result_path, osp.basename(experiment_path))
    print('* Saving predictions to ' + save_results_path)
    times = []
    for i in tqdm(range(len(test_dataset))):
        im_tens, mask, coords_crop, original_sz, im_name = test_dataset[i]
        start_time = time.perf_counter()
        full_pred = create_pred(model, im_tens, mask, coords_crop, original_sz, tta=tta)
        times.append(time.perf_counter() - start_time)
        save_pred(full_pred, save_results_path, im_name)

    print(f"* Average image time: {np.mean(times):g}s")
    print('* Done')

def get_args():

    parser = argparse.ArgumentParser()
    required_named = parser.add_argument_group('required arguments')
    required_named.add_argument('--dataset', type=str, help='generate results for which dataset', required=True)
    parser.add_argument('--split', type=str, default='', help='split to use')
    parser.add_argument('--tta', type=str, default='from_preds', help='test-time augmentation (no/from_logits/from_preds)')
    parser.add_argument('--config_file', type=str, default=None, help='experiments/name_of_config_file, overrides everything')
    # im_size overrides config file
    parser.add_argument('--im_size', help='delimited list input, could be 600,400', type=str, default='512')
    parser.add_argument('--device', type=str, default='cuda:0', help='where to run the training code (e.g. "cpu" or "cuda:0") [default: %(default)s]')
    parser.add_argument('--in_c', type=int, default=3, help='channels in input images')
    parser.add_argument('--use_green', type=int, default=0, help='if 0 and in_c=1, converts to gray. Use green channel otherwise')
    parser.add_argument('--result_path', type=str, default='../results', help='path to save predictions (defaults to results')

    return parser.parse_args()

if __name__ == '__main__':

    _args = get_args()
    main(_args)

