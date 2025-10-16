import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random
from PIL import Image

class lol:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self, parse_patches=True, validation='snow'):
        print("=> evaluating outdoor lol test set...")
        train_dataset = lolDataset(dir=os.path.join(self.config.data.data_dir, 'data', 'lol', 'train'),
                                        n=self.config.training.patch_n,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                        filelist=None,
                                        parse_patches=parse_patches)
        val_dataset = lolDataset(dir=os.path.join(self.config.data.data_dir, 'data', 'val'),
                                      n=self.config.training.patch_n,
                                      patch_size=self.config.data.image_size,
                                      transforms=self.transforms,
                                      filelist=None,
                                      parse_patches=parse_patches)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader

class lolDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, filelist=None, parse_patches=True):
        super().__init__()

        if filelist is None:
            lol_dir = dir
            input_names, gt_names, reflection_names, he_names, at_names = [], [], [], [], []

            # lol train filelist
            snow_inputs = os.path.join(lol_dir, 'input')
            images = [f for f in listdir(snow_inputs) if isfile(os.path.join(snow_inputs, f))]
            print(len(images))
            # assert len(images) == 485
            input_names += [os.path.join(snow_inputs, i) for i in images]
            gt_names += [os.path.join(os.path.join(lol_dir, 'gt'), i) for i in images]
            reflection_names += [os.path.join(os.path.join(lol_dir, 'reflection'), i) for i in images]
            he_names += [os.path.join(os.path.join(lol_dir, 'he'), i) for i in images]
            at_names += [os.path.join(os.path.join(lol_dir, 'at'), i) for i in images]
            print(len(input_names))

            x = list(enumerate(input_names))
            random.shuffle(x)
            indices, input_names = zip(*x)
            gt_names = [gt_names[idx] for idx in indices]
            reflection_names = [reflection_names[idx] for idx in indices]
            he_names = [he_names[idx] for idx in indices]
            at_names = [at_names[idx] for idx in indices]
            self.dir = None
        else:
            self.dir = dir
            train_list = os.path.join(dir, filelist)
            with open(train_list) as f:
                contents = f.readlines()
                input_names = [i.strip() for i in contents]
                gt_names = [i.strip().replace('input', 'gt') for i in input_names]

                reflection_names = [i.strip().replace('input', 'reflection') for i in input_names]
                he_names = [i.strip().replace('input', 'he') for i in input_names]
                at_names = [i.strip().replace('input', 'at_names') for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.reflection_names = reflection_names
        self.he_names = he_names
        self.at_names = at_names
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        reflection_name = self.reflection_names[index]
        he_name = self.he_names[index]
        at_names = self.at_names[index]

        img_id = re.split('/', input_name)[-1][:-4]
        input_img = PIL.Image.open(os.path.join(self.dir, input_name)) if self.dir else PIL.Image.open(input_name)
        reflection_img = PIL.Image.open(os.path.join(self.dir, reflection_name)) if self.dir else PIL.Image.open(reflection_name)
        he_img = PIL.Image.open(os.path.join(self.dir, he_name)) if self.dir else PIL.Image.open(he_name)
        input_at = PIL.Image.open(os.path.join(self.dir, at_names)) if self.dir else PIL.Image.open(at_names)
        try:
            gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)) if self.dir else PIL.Image.open(gt_name)
        except:
            gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)).convert('RGB') if self.dir else \
                PIL.Image.open(gt_name).convert('RGB')

        if self.parse_patches:
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n)
            input_img = self.n_random_crops(input_img, i, j, h, w)
            gt_img = self.n_random_crops(gt_img, i, j, h, w)
            reflection_img = self.n_random_crops(reflection_img, i, j, h, w)
            he_img = self.n_random_crops(he_img, i, j, h, w)
            input_at = self.n_random_crops(input_at, i, j, h, w)
            outputs = [torch.cat([self.transforms(input_img[i]), self.transforms(gt_img[i]), self.transforms(reflection_img[i]),
                                  self.transforms(he_img[i]), self.transforms(input_at[i])], dim=0)
                       for i in range(self.n)]
            return torch.stack(outputs, dim=0), img_id
        else:
            # Resizing images to multiples of 16 for whole-image restoration
            wd_new, ht_new = input_img.size
            if ht_new > wd_new and ht_new > 1024:
                wd_new = int(np.ceil(wd_new * 1024 / ht_new))
                ht_new = 1024
            elif ht_new <= wd_new and wd_new > 1024:
                ht_new = int(np.ceil(ht_new * 1024 / wd_new))
                wd_new = 1024
            wd_new = int(10 * np.ceil(wd_new / 10.0))
            ht_new = int(10 * np.ceil(ht_new / 10.0))
            input_img = input_img.resize((wd_new, ht_new), Image.Resampling.LANCZOS)
            gt_img = gt_img.resize((wd_new, ht_new), Image.Resampling.LANCZOS)
            reflection_img = reflection_img.resize((wd_new, ht_new), Image.Resampling.LANCZOS)
            he_img = he_img.resize((wd_new, ht_new), Image.Resampling.LANCZOS)
            input_at = input_at.resize((wd_new, ht_new), Image.Resampling.LANCZOS)

            return torch.cat([self.transforms(input_img), self.transforms(gt_img), self.transforms(reflection_img),
                              self.transforms(he_img), self.transforms(input_at)], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
