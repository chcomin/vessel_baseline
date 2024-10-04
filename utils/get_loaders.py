import random
import os.path as osp
import pandas as pd
from PIL import Image
import numpy as np
from skimage import measure
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from . import paired_transforms_tv04 as p_tr
from torchvision.transforms import v2 as tv_transf
from torchvision.transforms.v2 import functional as F
from torchvision import tv_tensors

class TrainDataset(Dataset):
    def __init__(self, csv_path, transforms=None, label_values=None):
        df = pd.read_csv(csv_path)
        self.im_list = df.im_paths
        self.gt_list = df.gt_paths
        self.mask_list = df.mask_paths
        self.transforms = transforms
        self.conversion = None
        self.label_values = label_values  # for use in label_encoding

    def label_encoding(self, gdt):
        gdt_gray = np.array(gdt.convert('L'))
        classes = np.arange(len(self.label_values))
        for i in classes:
            gdt_gray[gdt_gray == self.label_values[i]] = classes[i]
        return Image.fromarray(gdt_gray)

    def crop_to_fov(self, img, target, mask):
        minr, minc, maxr, maxc = measure.regionprops(np.array(mask))[0].bbox
        im_crop = Image.fromarray(np.array(img)[minr:maxr, minc:maxc])
        tg_crop = Image.fromarray(np.array(target)[minr:maxr, minc:maxc])
        mask_crop = Image.fromarray(np.array(mask)[minr:maxr, minc:maxc])
        return im_crop, tg_crop, mask_crop

    def __getitem__(self, index):
        # load image and labels
        img = Image.open(self.im_list[index])
        target = Image.open(self.gt_list[index])
        mask = Image.open(self.mask_list[index]).convert('L')

        img, target, mask = self.crop_to_fov(img, target, mask)

        target = self.label_encoding(target)

        target = np.array(self.label_encoding(target))

        target[np.array(mask) == 0] = 0
        target = Image.fromarray(target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # QUICK HACK FOR PSEUDO_SEG IN VESSELS, BUT IT SPOILS A/V
        if len(self.label_values)==2: # vessel segmentation case
            target = target.float()
            if torch.max(target) >1:
                target= target.float()/255

        return img, target

    def __len__(self):
        return len(self.im_list)

class TestDataset(Dataset):
    def __init__(self, csv_path, tg_size):
        df = pd.read_csv(csv_path)
        self.im_list = df.im_paths
        self.mask_list = df.mask_paths
        self.tg_size = tg_size

    def crop_to_fov(self, img, mask):
        mask = np.array(mask).astype(int)
        minr, minc, maxr, maxc = measure.regionprops(mask)[0].bbox
        im_crop = Image.fromarray(np.array(img)[minr:maxr, minc:maxc])
        return im_crop, [minr, minc, maxr, maxc]

    def __getitem__(self, index):
        # # load image and mask
        img = Image.open(self.im_list[index])
        mask = Image.open(self.mask_list[index]).convert('L')
        img, coords_crop = self.crop_to_fov(img, mask)
        original_sz = img.size[1], img.size[0]  # in numpy convention

        img = F.pil_to_tensor(img)
        img = F.resize(img, self.tg_size)
        img = F.to_dtype(img, torch.float32, scale=True)

        return img, np.array(mask).astype(bool), coords_crop, original_sz, self.im_list[index]

    def __len__(self):
        return len(self.im_list)

def get_train_val_datasets(csv_path_train, csv_path_val, tg_size=(512, 512), label_values=(0, 255)):

    train_dataset = TrainDataset(csv_path=csv_path_train, label_values=label_values)
    val_dataset = TrainDataset(csv_path=csv_path_val, label_values=label_values)
    # transforms definition
    # required transforms
    def train_t(img, target):

        scale = tv_transf.RandomAffine(degrees=0, scale=(0.95, 1.20))
        transl = tv_transf.RandomAffine(degrees=0, translate=(0.05, 0))
        rotate = tv_transf.RandomRotation(degrees=45)
        scale_transl_rot = [scale, transl, rotate]
        brightness, contrast, saturation, hue = 0.25, 0.25, 0.25, 0.01
        jitter = tv_transf.ColorJitter(brightness, contrast, saturation, hue)

        img = F.pil_to_tensor(img)
        target = F.pil_to_tensor(target)

        img = F.resize(img, tg_size)
        target = F.resize(target, tg_size, interpolation=tv_transf.InterpolationMode.NEAREST_EXACT)

        idx = random.randint(0,2)
        t = scale_transl_rot[idx]
        params = t._get_params([img])
        if idx==2:
            img = F.rotate(img, **params,  interpolation=tv_transf.InterpolationMode.BILINEAR, fill=0)
            target = F.rotate(target, **params, fill=0)
        else:
            img = F.affine(img, **params, fill=0)
            target = F.affine(target, **params, fill=0)
        img = jitter(img)
        
        if random.random() < 0.5:
            img, target = F.hflip(img), F.hflip(target)
        if random.random() < 0.5:
            img, target = F.vflip(img), F.vflip(target)

        img = F.to_dtype(img, torch.float32, scale=True)
        target = F.to_dtype(target, torch.int64, scale=False).mul(255)[0]

        return img, target
    
    def val_t(img, target):

        img = F.resize(img, tg_size)
        target = F.resize(target, tg_size, interpolation=tv_transf.InterpolationMode.NEAREST)      
        img = F.pil_to_tensor(img)
        target = F.pil_to_tensor(target)[0]

        img = F.to_dtype(img, torch.float32, scale=True)
        target = F.to_dtype(target, torch.int64, scale=False).mul(255)

        return img, target

    train_dataset.transforms = train_t
    val_dataset.transforms = val_t

    return train_dataset, val_dataset

def get_train_val_loaders(csv_path_train, csv_path_val, batch_size=4, tg_size=(512, 512), label_values=(0, 255), num_workers=0):
    train_dataset, val_dataset = get_train_val_datasets(csv_path_train, csv_path_val, tg_size=tg_size, label_values=label_values)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(), shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader

def get_test_dataset(data_path, csv_path='test.csv', tg_size=(512, 512)):
    # csv_path will only not be test.csv when we want to build training set predictions
    path_test_csv = osp.join(data_path, csv_path)
    test_dataset = TestDataset(csv_path=path_test_csv, tg_size=tg_size)

    return test_dataset



