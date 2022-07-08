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
import json

# MODULE IMPORTS
import numpy as np
# import pandas as pd
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

from utils import *


# FastAI Learner + Pytorch model definitions

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    print('Loading Database...', end='\t')
    datab = fastai.basic_train.load_data(
        'C:/Users/chris/Downloads/training_cov_npy',
        'datab.pkl',
        collate_fn=pad_collate,
        pin_memory=False,
        num_workers=4,
        bs=512
    )
    print('Done')
    
    # Stratified K-Fold Crossval
    learners = []
    skf = StratifiedKFold(10, random_state=42, shuffle=True)
    if os.path.exists('prototyping/model.json'):
        with open('prototyping/model.json', 'r') as f:
            interruptable_info = json.load(f)
    else:
        interruptable_info = {}
        interruptable_info['fold'] = 0
        interruptable_info['epoch'] = 0
    for idx, (train_idx, test_idx) in enumerate(skf.split(range(len(datab.test_ds.x)), np.any(datab.test_ds.y.items, axis=1).astype('int'))):
        if idx >= interruptable_info['fold']:
            print("%d/10 Folds" % (idx+1))
            split_ils = datab.test_ds.x.split_by_idxs(train_idx, test_idx)
            split_ils_y = datab.test_ds.y.split_by_idxs(train_idx, test_idx)
            x_train, x_valid, y_train, y_valid = split_ils.train, split_ils.valid, \
                split_ils_y.train, split_ils_y.valid
            ll = fastai.data_block.LabelLists(
                datab.path,
                train=data_block.LabelList(x_train, y_train, tfms=[]),
                valid=data_block.LabelList(x_valid, y_valid, tfms=[]),
            )
            ll.test = datab.test_ds
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
                [32] * 9,
                2,
                0.2,
                use_skip_connections=True,
                reduce_dimensionality=False
            )

            learn = Learner(data=datab, model=net.cuda())

            # bce_weights = np.ones(336, dtype=float) * 40336.0
            # bce_weights[-10:] = [21.570053, 15.13546,  13.846893, 13.827906, 13.804244, 13.804244, 13.794802, 13.794802, 13.780663, 13.757162]
            # bce_weights = torch.tensor(bce_weights, device=torch.device('cuda:0'))

            if interruptable_info['epoch'] != 0 and os.path.exists('prototyping/10fold-100(2)/ProtoTCN-%d_fold-current.pth' % (idx+1)):
                learn = Learner.load('prototyping/10fold-100(2)/ProtoTCN-%d_fold-current_%d.pth' % (idx+1, interruptable_info['epoch']+1))
            else:
                learn = Learner(
                    data=datab,
                    model=net.cuda(),
                    loss_func=nn.BCEWithLogitsLoss(
                        # pos_weight=bce_weights
                    ),
                    path='prototyping',
                    model_dir='10fold-100(2)',
                    callback_fns=[
                        partial(
                            fastai.callbacks.CSVLogger, filename="%d-fold_history" % (idx+1),
                            append=True
                            ),
                        ShowGraph
                    ],
                    metrics=[partial(fbeta, beta=14.7)],
                )
            # learn.fit(
                # 1, 3e-4,
                # callbacks=[
                    # fastai.callbacks.TerminateOnNaNCallback(),
                # #     fastai.callbacks.SaveModelCallback(
                # #         learn, every='improvement', monitor='loss', name='proto'
                    # # )
                # ],
            # )
            learn.fit_one_cycle(
                cyc_len=100, max_lr=7e-4, wd=1e-3, start_epoch=interruptable_info['epoch'], callbacks=[
                    TerminateOnNaNCallback(),
                    SaveModelCallback(
                        learn, every='improvement', monitor='valid_loss', name='ProtoTCN-%d_fold-best' % (idx+1)
                    ),
                    SaveModelCallback(
                        learn, every='epoch', monitor='valid_loss', name='ProtoTCN-%d_fold-current' % (idx+1)
                    ),
                    JSONLogger(learn, idx, 'model.json')
                ]
            )
            interruptable_info['epoch'] = 0
