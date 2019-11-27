import numpy as np

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

from TCN.TCN.tcn import TemporalConvNet, TemporalSkipBlock

class patient_data(core.ItemBase):
    def __init__(self, fn):
#         f = np.memmap(fn, dtype='float64', mode='c')
#         f = f.reshape(f[-3:].astype('int32'))
        f = np.load(fn, allow_pickle=True)
        self.obj = f
        self.len = f.shape[-1]
        self.data = np.vstack((f[:, 0, :], f[:6, -1, :]))
        means, covs = [], []
        self.idxs = []
        self.sepsis = f[6, -1, :]
#         start = time.time()
#         timings = []
        for i, feature in enumerate(f):
            if feature[1:self.len+1, :].any() and len(feature[1:self.len+1, :].shape) == 2:
                self.idxs.append(i)
                means.append(feature[0, :])
                covs.append(feature[1:self.len+1, :])
#         timings.append(time.time()-start)
        self.data = torch.tensor(self.data)
#         timings.append(time.time()-start)
        if len(covs) > 0 and len(means) > 0:
            self.N = torch.distributions.MultivariateNormal(
                torch.tensor(means),
                torch.tensor(covs)
            )
        else:
            self.N = None
#         timings.append(time.time()-start)
#         print(timings)
#         self.data = F.pad(self.data, ((336-self.len), 0))
    def __len__(self):
        return self.len

    def __str__(self):
        return "{} [{}]".format(self.data.shape, self.sepsis.any())
        
    def gaussian_sample(self):
        data = self.data
        if self.N is not None:
            sample = self.N.sample()
            for sample_idx, tensor_idx in enumerate(self.idxs):
#                 data[tensor_idx, 336-self.len:] = sample[sample_idx]
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
#         f = np.memmap(filename, dtype='float64', mode='c')
#         f = f.reshape(f[-3:].astype('int32'))
#         f = np.array(f)
        return patient_data(filename)

    def reconstruct(self, t):
        return

def zero_padder(fn):
    f = np.load(fn, mmap_mode='c', allow_pickle=True)
    l = f.shape[-1]
    sepsis_label = f[6, -1, :]
    sepsis_label = np.pad(sepsis_label, ((336-l), 0), mode='constant')
    print("%s: %s [%d, %s]   " % (fn, sepsis_label, l, sepsis_label.any()), end='\r')
    return sepsis_label

def pad_collate(batch):
    data = torch.stack([F.pad(
        item[0].data,
        ((336-len(item[0])), 0)
    ) for item in batch], dim=0)
    targets = torch.stack([torch.from_numpy(item[1].data) for item in batch], dim=0)
    return [data, targets]

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
        x = x.to(self.network[0].net[0].weight.dtype)
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
        y1 = self.tcn(x)
        o = self.output(y1.transpose(1, 2)).squeeze()
        return o