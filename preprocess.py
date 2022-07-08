import sys
import os
import collections
import functools
import asyncio
from pathlib import Path
import traceback

from utils import *

# Modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn

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

dl = data_list.from_folder('C:/Users/chris/Downloads/training_cov_npy')
dl.filter_by_func(lambda fn: fn.suffix == ".npy")

sd = dl.split_none()
ll = sd.label_from_func(
    zero_padder,
    classes=range(336),
    label_cls=data_block.MultiCategoryList,
    one_hot=True
)
ll.test = ll.train
datab = ll.databunch(
    collate_fn=pad_collate,
    num_workers=1,
    bs=1024
)
datab.save('datab.pkl')