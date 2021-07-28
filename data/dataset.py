# -*- coding: utf-8 -*-
import os
import time
import torch
import itertools
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold

import batchgenerators.transforms as DA
# from batchgenerators.transforms.crop_and_pad_transform import CenterCropTransform


class CTdataset(Dataset):
    def __init__(self, indexes, image_data, descriptor, root_dir, patch_size):
        super().__init__()
        self.root_dir = root_dir #+ '/Processed'
        self.patch_size = np.asarray(patch_size)
        self.task = -2  # -2=cáncer; -1=nodulos/masas

        labels = pd.read_csv(image_data)
        labels = labels[labels['Número de acceso'].isin(indexes)]
        #labels = labels.iloc[indexes, :]
        self.labels = labels.set_index('Número de acceso',
                                       drop=True).T.to_dict('list')
        desc = pd.read_csv(descriptor).astype({'Número de acceso': int})
        desc = desc.set_index('Número de acceso', drop=True).T
        self.descriptor = desc.iloc[np.r_[1:12,
                                    13:desc.shape[0]-1], :].to_dict('list')
        self.idx = list(self.labels.keys())
        #sizes = pd.read_csv('/media/bcv007/SSD7/ladaza/SSD/Data/fsfb/Lucas/Data/sizes.csv')
        #sizes = sizes.set_index('access').T.to_dict('list')
        # for i in sizes:
        #     if int(sizes[i][0]) < 250 and i in self.idx:
        #         self.idx.remove(i)
        self.weights = self.weights_balanced()

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        patient = self.idx[idx]
        label = self.labels[patient][self.task]
        descriptor = self.descriptor[patient]
        
        descriptor = np.array(descriptor, dtype=np.float32)
            
        image, _ = load_image(str(patient) + '.nii.gz', self.root_dir)

        if any(np.asarray(image.shape) <= self.patch_size):
            dif = self.patch_size - image.shape
            mod = dif % 2
            dif = dif // 2
            pad = np.maximum(dif, [0, 0, 0])
            pad = tuple(zip(pad, pad + mod))
            image = np.pad(image, pad, 'reflect')

        sz = self.patch_size[0]
        if any(np.asarray(image.shape) >= self.patch_size):
            x, y, z = image.shape
            x = x // 2 - (sz // 2)
            y = y // 2 - (sz // 2)
            z = z // 2 - (sz // 2)
            image = image[x:x + sz, y:y + sz, z:z + sz]
        # Stats obtained from the MSD dataset
        image = np.clip(image, a_min=-1024, a_max=326)
        image = (image - 159.14433291523548) / 323.0573880113456

        return {'data': np.expand_dims(image, 0),
                'descriptor': descriptor, 'target': label, 'id': patient}

    def weights_balanced(self):
        count = [0] * 2
        for item in self.idx:
            count[self.labels[item][self.task]] += 1
        weight_per_class = [0.] * 2
        N = float(sum(count))
        for i in range(2):
            weight_per_class[i] = N/float(count[i])
        weight = [0] * len(self.idx)
        for idx, val in enumerate(self.idx):
            weight[idx] = weight_per_class[self.labels[val][self.task]]
        return weight

class DescDataset(Dataset):
    def __init__(self, indexes, image_data, descriptor, root_dir, patch_size):
        super().__init__()
        
        self.root_dir = root_dir #+ '/Processed'
        self.patch_size = np.asarray(patch_size)
        self.task = -2  # -2=cáncer; -1=nodulos/masas

        labels = pd.read_csv(image_data)
        labels = labels[labels['Número de acceso'].isin(indexes)]
        #labels = labels.iloc[indexes, :]
        self.labels = labels.set_index('Número de acceso',
                                       drop=True).T.to_dict('list')
        desc = pd.read_csv(descriptor).astype({'Número de acceso': int})
        desc = desc.set_index('Número de acceso', drop=True).T
        self.descriptor = desc.iloc[np.r_[1:12,
                                    13:desc.shape[0]-1], :].to_dict('list')
        self.idx = list(self.labels.keys())
        
        self.weights = self.weights_balanced()
        
    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        patient = self.idx[idx]
        label = self.labels[patient][self.task]
        descriptor = self.descriptor[patient]
        descriptor = np.array(descriptor, dtype=np.float32)
        
        return {'data': np.zeros(1), 'descriptor': descriptor, 'target': label, 'id': patient}
    
    def weights_balanced(self):
        count = [0] * 2
        for item in self.idx:
            count[self.labels[item][self.task]] += 1
        weight_per_class = [0.] * 2
        N = float(sum(count))
        for i in range(2):
            weight_per_class[i] = N/float(count[i])
        weight = [0] * len(self.idx)
        for idx, val in enumerate(self.idx):
            weight[idx] = weight_per_class[self.labels[val][self.task]]
        return weight
class collate(object):
    def __init__(self, size, data_aug=False):
        # rot_angle = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        if data_aug:
            self.transforms = DA.Compose([
                DA.ContrastAugmentationTransform((0.3, 3.), preserve_range=True),
                DA.GammaTransform((0.7, 1.5), invert_image=False, per_channel=True,
                                retain_stats=True, p_per_sample=0.3), 
                DA.NumpyToTensor()])
        else: 
            self.transforms = DA.Compose([DA.NumpyToTensor()])

    def __call__(self, batch):
        elem = batch[0]
        batch = {key: np.stack([d[key] for d in batch]) for key in elem}
        return self.transforms(**batch)


def test_voxels(patch_size, im_shape):
    center = patch_size // 2
    dims = []
    for i, j in zip(im_shape, center):
        end = i - j
        num = np.ceil((end - j) / j)
        if num == 1:
            num += 1
        voxels = np.linspace(j, end, int(num))
        dims.append(voxels)
    voxels = list(itertools.product(*dims))
    return voxels


def load_image(patient, root_dir):
    im = nib.load(os.path.join(root_dir, patient))
    image = im.get_fdata()
    return image, im.affine


def extract_patch(image, voxel, patch_size):
    im_size = image.shape
    v1 = np.asarray(voxel) - patch_size // 2
    v1 = v1.astype(int)
    v2 = np.minimum(v1 + patch_size, im_size)

    patch_list = []
    patch = image[v1[0]:v2[0], v1[1]:v2[1], v1[2]:v2[2]]
    patch = verify_size(patch, patch_size)
    patch_list.append(patch)
    return np.stack(patch_list, axis=0)


def verify_size(im, size):
    dif = np.asarray(size) - im.shape
    if any(dif > 0):
        dif[dif < 0] = 0
        mod = dif % 2
        dif = np.abs(dif) // 2
        pad = tuple(zip(dif, dif + mod))
        im = np.pad(im, pad, 'reflect')
    return im


def save_image(prediction, outpath, affine):
    new_pred = nib.Nifti1Image(prediction.numpy(), affine)
    new_pred.set_data_dtype(np.uint8)
    nib.save(new_pred, outpath)

def kfold_indexes(file_path, folds=2, task=-2):
    file = pd.read_csv(file_path)
    k_indexes = StratifiedKFold(n_splits=folds)
    k_indexes = list(k_indexes.split(file['Número de acceso'], file.iloc[:, task]))

    return k_indexes
