import os
import os.path as osp
import pandas as pd
from PIL import Image
import numpy as np
from skimage.measure import regionprops
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from . import paired_transforms_tv04 as p_tr

class TrainDataset(Dataset):
    def __init__(self, csv_path, transforms=None, label_values=None):
        df = pd.read_csv(csv_path)
        self.im_list = df.im_paths
        self.gt_list = df.gt_paths
        self.mask_list = df.mask_paths
        self.transforms = transforms
        self.label_values = label_values  # for use in label_encoding

    def label_encoding(self, gdt):
        gdt_gray = np.array(gdt.convert('L'))
        classes = np.arange(len(self.label_values))
        for i in classes:
            gdt_gray[gdt_gray == self.label_values[i]] = classes[i]
        return Image.fromarray(gdt_gray)

    def crop_to_fov(self, img, target, mask):
        minr, minc, maxr, maxc = regionprops(np.array(mask))[0].bbox
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
        minr, minc, maxr, maxc = regionprops(mask)[0].bbox
        im_crop = Image.fromarray(np.array(img)[minr:maxr, minc:maxc])
        return im_crop, [minr, minc, maxr, maxc]

    def __getitem__(self, index):
        # # load image and mask
        img = Image.open(self.im_list[index])
        mask = Image.open(self.mask_list[index]).convert('L')
        img, coords_crop = self.crop_to_fov(img, mask)
        original_sz = img.size[1], img.size[0]  # in numpy convention

        # # load image and mask
        # img = Image.open(self.im_list[index])
        # original_sz = img.size[1], img.size[0]  # in numpy convention
        # mask = Image.open(self.mask_list[index]).convert('L')
        # img, coords_crop = self.crop_to_fov(img, mask)
        # print(self.im_list[index], 'original size inside dataset', original_sz)

        rsz = p_tr.Resize(self.tg_size)
        tnsr = p_tr.ToTensor()
        tr = p_tr.Compose([rsz, tnsr])
        img = tr(img)  # only transform image

        return img, np.array(mask).astype(bool), coords_crop, original_sz, self.im_list[index]

    def __len__(self):
        return len(self.im_list)

def get_train_val_datasets(csv_path_train, csv_path_val, tg_size=(512, 512), label_values=(0, 255)):

    train_dataset = TrainDataset(csv_path=csv_path_train, label_values=label_values)
    val_dataset = TrainDataset(csv_path=csv_path_val, label_values=label_values)
    # transforms definition
    # required transforms
    resize = p_tr.Resize(tg_size)
    tensorizer = p_tr.ToTensor()
    # geometric transforms
    h_flip = p_tr.RandomHorizontalFlip()
    v_flip = p_tr.RandomVerticalFlip()
    rotate = p_tr.RandomRotation(degrees=45, fill=0, fill_tg=(0,))
    scale = p_tr.RandomAffine(degrees=0, scale=(0.95, 1.20))
    transl = p_tr.RandomAffine(degrees=0, translate=(0.05, 0))
    # either translate, rotate, or scale
    scale_transl_rot = p_tr.RandomChoice([scale, transl, rotate])
    # intensity transforms
    brightness, contrast, saturation, hue = 0.25, 0.25, 0.25, 0.01
    jitter = p_tr.ColorJitter(brightness, contrast, saturation, hue)
    train_transforms = p_tr.Compose([resize,  scale_transl_rot, jitter, h_flip, v_flip, tensorizer])
    val_transforms = p_tr.Compose([resize, tensorizer])
    train_dataset.transforms = train_transforms
    val_dataset.transforms = val_transforms

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



