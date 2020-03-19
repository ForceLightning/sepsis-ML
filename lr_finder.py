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
        'D:/training_cov_npy/',
        'datab.pkl',
        collate_fn=pad_collate,
        pin_memory=False,
        num_workers=4,
        bs=1024
    )
    x, y = datab.test_ds.x, datab.test_ds.y
    ll = fastai.data_block.LabelLists(
        datab.path,
        train=data_block.LabelList(x, y, tfms=[]),
        valid=data_block.LabelList(x, y, tfms=[])
    )
    datab = ll.transform([]).databunch(
        collate_fn=pad_collate,
        num_workers=4,
        pin_memory=True,
        bs=256
    )
    print('Done')
    
    # Stratified K-Fold Crossval
    for wd in tqdm([0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]):
        net = TCN(
            40,
            336,
            [32] * 9,
            2,
            0.2,
            use_skip_connections=True,
            reduce_dimensionality=False
        )

        learn = Learner(
            data=datab,
            model=net.cuda(),
            loss_func=nn.BCEWithLogitsLoss(),
        )
        learn.lr_find(num_it=500, stop_div=True, wd=wd)
        fig = learn.recorder.plot(suggestion=True, return_fig=True)
        fig.savefig("tcn-9_wd-%0.2f.png" % (wd), dpi=300, bbox_inches='tight')
        print(wd, learn.recorder.min_grad_lr)