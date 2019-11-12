# SYSTEM IMPORTS
import sys
import os
import collections
import functools
import time
from pathlib import Path
import traceback
from multiprocessing import Pool
from functools import partial

# MODULE IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import h5py
import sklearn

## Specific module imports
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel)
from scipy.interpolate import interp1d, UnivariateSpline
from sklearn.model_selection import StratifiedKFold
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from tqdm.autonotebook import tqdm

## FastAI and Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

import fastai
from fastai.basic_train import *
from fastai.train import *
from fastai.metrics import *
from fastai.callbacks import *
from fastai import core, data_block
from fastai.distributed import *

from TCN.TCN.tcn import TemporalConvNet, TCN_DimensionalityReduced, TemporalSkipBlock

# FastAI Datablock API

class patient_data(core.ItemBase):
    def __init__(self, fn):
        f = np.load(fn, allow_pickle=True)
        self.obj = f
        self.len = f.shape[-1]
        self.data = np.vstack((f[:, 0, :], f[:6, -1, :]))
        means, covs = [], []
        self.idxs = []
        self.sepsis = f[6, -1, :]
        for i, feature in enumerate(f):
            if feature[1:self.len+1, :].any() and len(feature[1:self.len+1, :].shape) == 2:
                self.idxs.append(i)
                means.append(feature[0, :])
                covs.append(feature[1:self.len+1, :])
        self.data = torch.tensor(self.data)
        if len(covs) > 0 and len(means) > 0:
            self.N = torch.distributions.MultivariateNormal(
                torch.tensor(means),
                torch.tensor(covs)
            )
        else:
            self.N = None
    def __len__(self):
        return self.len

    def __str__(self):
        return "{} [{}]".format(self.data.shape, self.sepsis.any())
        
    def gaussian_sample(self):
        data = self.data
        if self.N is not None:
            sample = self.N.sample()
            for sample_idx, tensor_idx in enumerate(self.idxs):
                data[tensor_idx] = sample[sample_idx]
        return data
    
    def apply_tfms(self, tfms, **kwargs):
        self.data = self.gaussian_sample()
        return self

class data_list(data_block.ItemList):
    def __init__(self, items, *args, **kwargs):
        super().__init__(items, **kwargs)
    
    def get(self, i):
        filename = super().get(i)
        return patient_data(filename)

    def reconstruct(self, t):
        return



def pad_collate(batch):
    data = torch.stack([F.pad(
        item[0].data,
        ((336-item[0].data.shape[1]), 0)
    ) for item in batch], dim=0)
    targets = torch.stack([torch.from_numpy(item[1].data) for item in batch], dim=0)
    return [data, targets]

# with Pool() as p:
#     ll = sd._label_from_list(
# #         [func(o) for o in sd.items],
#         list(tqdm(p.imap(zero_padder, sd.items), total=len(sd.items))),
#         label_cls=data_block.MultiCategoryList,
#     )

# datab = ll.transform([]).databunch(
#     collate_fn=pad_collate,
#     num_workers=1,
# #     pin_memory=True,
#     bs=1024
# )
# datab.save('datab_itembaseloading.pkl')



# FastAI Learner + Pytorch model definitions

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    dl = data_list.from_folder('D:/training_cov_npy/')
    dl.filter_by_func(lambda fn: fn.suffix == '.npy')
    sd = dl.split_none()

    def zero_padder(fn):
        f = np.load(fn, mmap_mode='c', allow_pickle=True)
        l = f.shape[-1]
        sepsis_label = f[6, -1, :]
        sepsis_label = np.pad(sepsis_label, ((336-l), 0), mode='constant')
        print("%s: %s [%d, %s]   " % (fn, sepsis_label, l, sepsis_label.any()), end='\r')
        return sepsis_label

    ll = sd.label_from_func(
        zero_padder,
        classes=range(336),
        label_cls=data_block.MultiCategoryList,
        one_hot=True
    )
    ll.test = ll.train
    ll.train = None
    print('\n')

    # print('Loading Database...', end='\t')
    # datab = fastai.basic_train.load_data(
    #     'challenge/interpolated/training_cov_npy',
    #     'datab.pkl',
    #     collate_fn=pad_collate,
    #     pin_memory=False,
    #     num_workers=8,
    # #     timeout=1000000
    #     bs=512
    # )
    # print('Done')

    class TCN_DimensionalityReduced(nn.Module):
        def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, use_skip_connections=False, reduce_dimensionality=True):
            super(TCN_DimensionalityReduced, self).__init__()
            self.use_skip_connections = use_skip_connections
            layers = []
            num_levels = len(num_channels)
            self.reduce_dimensionality = reduce_dimensionality
            if self.reduce_dimensionality:
                self.d_reduce = nn.Conv1d(num_inputs, num_channels[0], kernel_size=1)
            for i in range(num_levels):
                dilation_size = 2 ** i
                in_channels = num_channels[i-1] if (self.reduce_dimensionality or i!=0) else num_inputs
                out_channels = num_channels[i]
                layers += [TemporalSkipBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                            padding=(kernel_size-1) * dilation_size, dropout=dropout)]
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            x = x.float()
            if self.use_skip_connections:
                for tb in [m for m in self.network.modules()][2].children():
                    skips = [layer for layer in tb.children()][-1]
                    self.network(x).add_(skips)
            if self.reduce_dimensionality:
                x = self.d_reduce(x)
            return self.network(x)

    class TCN(nn.Module):
        def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, **kwargs):
            super(TCN, self).__init__()
            self.tcn = TCN_DimensionalityReduced(input_size,
                                                num_channels,
                                                kernel_size=kernel_size,
                                                dropout=dropout,
                                                **kwargs)
            self.output = nn.Linear(num_channels[-1], 1)
            self.init_weights()

        def init_weights(self):
            self.output.weight.data.normal_(0, 0.01)

        def forward(self, x):
            x = x.to(self.tcn.network[0].net[0].weight.dtype)
            y1 = self.tcn(x)
            o = self.output(y1.transpose(1, 2)).squeeze()
            return o

    # Stratified K-Fold Crossval
    learners = []
    skf = StratifiedKFold(10, random_state=42)
    for idx, (train_idx, test_idx) in enumerate(skf.split(range(len(ll.test.x)), np.any(ll.test.y.items, axis=1).astype('int'))):
        print("%d/10 Folds" % (idx+1))
        split_ils = ll.test.x.split_by_idxs(train_idx, test_idx)
        split_ils_y = ll.test.y.split_by_idxs(train_idx, test_idx)
        x_train, x_valid, y_train, y_valid = split_ils.train, split_ils.valid, \
            split_ils_y.train, split_ils_y.valid
        ll.train = data_block.LabelList(x_train, y_train, tfms=[])
        ll.valid = data_block.LabelList(x_valid, y_valid, tfms=[])
        print("Loading Database", end='\t')
        datab = ll.transform([]).databunch(
            collate_fn=pad_collate,
            num_workers=4,
            pin_memory=True,
            bs=512
        )
        print("Done")


        net = TCN(
            40,
            336,
            [32] * 3,
            64,
            0.2,
            use_skip_connections=True,
            reduce_dimensionality=False
        )

        learn = Learner(
            data=datab,
            model=net.cuda(),
            loss_func=nn.BCEWithLogitsLoss(),
            path='prototyping',
            model_dir='fold-%d' % (idx+1),
            callback_fns=[
                fastai.callbacks.CSVLogger,
                ShowGraph
            ],
            metrics=[fbeta],
        ).to_fp16(loss_scale=512)
        learn.fit(
            1,
            callbacks=[
                fastai.callbacks.TerminateOnNaNCallback(),
            #     fastai.callbacks.SaveModelCallback(
            #         learn, every='improvement', monitor='loss', name='proto'
                # )
            ],
        )
        learners.append(learn)