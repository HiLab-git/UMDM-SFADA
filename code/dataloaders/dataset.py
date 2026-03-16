# -*- coding: utf-8 -*-

from logging import root
import os
from scipy import ndimage
import torch
import random
import h5py
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import itertools
from torchvision import transforms
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2
import skimage
from torchvision.transforms import Compose
class threeDDataSet(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            num=None,
            transform=None,
            ops_weak=None,
            ops_strong=None,
            active_method=None
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        self.active_method = active_method
        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/trainlist.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "train_stage1" and self.active_method:
            with open(self._base_dir + f"/stage1_{self.active_method}.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "train_stage2" and self.active_method:
            with open(self._base_dir + f"/all_{self.active_method}.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "semi_train" and self.active_method:
            with open(self._base_dir + f"/all_{self.active_method}.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "semi_train":
            with open(self._base_dir + "/all.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "val":
            with open(self._base_dir + "/vallist.txt", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train" or "train_stage1" or "train_stage2" or '1' or '2' or '3' or '4':
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "val":
            h5f = h5py.File(self._base_dir + "/{}".format(case), "r")
        elif self.split == "train":
            h5f = h5py.File(self._base_dir + "/{}".format(case), "r")
        elif self.split == "semi_train":
            h5f = h5py.File(self._base_dir + "/{}".format(case), "r")

        image = h5f["image"][:]
        label = h5f["label"][:]
        image = image.astype(np.float32)    
        label = label.astype(np.uint8)
        sample = {"image": image, "label": label}
        if (self.split == "train" or self.split == "semi_train"):
            sample["image"] = torch.from_numpy(sample["image"]).float().unsqueeze(0)
            sample["label"] = torch.from_numpy(sample["label"]).to(torch.int16).unsqueeze(0)
            sample_data = {"image": sample["image"], "label": sample["label"]}
            if self.transform is not None:
                sample_data = self.transform(**sample_data)
            sample["image"] = sample_data["image"]
            sample["label"] = sample_data["label"]
        sample["idx"] = case
        return sample

class twoDDataSet(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            num=None,
            transform=None,
            ops_weak=None,
            ops_strong=None,
            active_method=None
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        self.active_method = active_method
        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "train_stage1" and self.active_method:
            with open(self._base_dir + f"/stage1_{self.active_method}.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "train_stage2" and self.active_method:
            with open(self._base_dir + f"/all_slice_{self.active_method}.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "semi_train" and self.active_method:
            with open(self._base_dir + f"/all_slice_{self.active_method}.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "semi_train":
            with open(self._base_dir + "/all.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "val":
            with open(self._base_dir + "/vallist.txt", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train" or "train_stage1" or "train_stage2" or '1' or '2' or '3' or '4':
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "val":
            h5f = h5py.File(self._base_dir + "/{}".format(case), "r")
        elif self.split == "train":
            h5f = h5py.File(self._base_dir + "/slices/{}".format(case), "r")
        elif self.split == "semi_train":
            h5f = h5py.File(self._base_dir + "/slices/{}".format(case), "r")

        image = h5f["image"][:]
        label = h5f["label"][:]
        image = image.astype(np.float32)    
        label = label.astype(np.uint8)
        sample = {"image": image, "label": label}
        if (self.split == "train" or self.split == "semi_train"):
            sample["image"] = torch.from_numpy(sample["image"]).float().unsqueeze(0)
            sample["label"] = torch.from_numpy(sample["label"]).to(torch.int16).unsqueeze(0)
            sample_data = {"image": sample["image"], "label": sample["label"]}
            sample_data = self.transform(**sample_data)
            sample["image"] = sample_data["image"]
            sample["label"] = sample_data["label"]
        sample["idx"] = case
        return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
# [5]  Zhou Z, Qi L, Yang X, et al. Generalizable cross-modality medical image segmentation via style augmentation and dual normalization[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 20856-20865.

import numpy as np
import random
import matplotlib.pyplot as plt
try:
    from scipy.special import comb
except:
    from scipy.misc import comb
    
"""  
this is for none linear transformation


"""

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def nonlinear_transformation(x):
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xvals, yvals = bezier_curve(points, nTimes=1000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x

def visualize_sample(sample):
    image = sample["image"][0].numpy()  # Assuming the batch size is 1
    label = sample["label"][0].numpy()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image[0], cmap="gray")
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    if len(label.shape) == 2:
        plt.imshow(label, cmap="gray")
    elif len(label.shape) == 3:
        plt.imshow(label[0], cmap="gray")  # Assuming label is 3D (batch_size, height, width)
    else:
        raise ValueError("Invalid shape for label data")

    plt.title("Original Label")

    plt.show()


if __name__ == "__main__":

    dataset = threeDDataSet(
        base_dir="/data1/lnd/SFDA/data/BC_AAI",
        split="val"
    )

    sample = dataset[1]
    print("Transformed Image Shape:", sample['image'].shape)
    print("Label Shape:", sample['label'].shape)
    print("Image Name", sample["idx"])
