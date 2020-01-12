import os
import json

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

class JSONLogger(LearnerCallback):
    """A `LearnerCallback` that saves k-fold and epoch values"""
    def __init__(self, learn:Learner, fold:int, filename:str = 'model.json', truncate:bool = True):
        super().__init__(learn)
        self.filename, self.path, self.fold, self.truncate = filename, self.learn.path/f'{filename}', fold, truncate

    def on_train_begin(self, **kwargs):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(self.path):
            with open(self.path, 'r+') as f:
                self.json = json.load(f)
                f.truncate(0)
                self.json['fold'] = self.fold
                json.dump(self.json, f, indent=4, sort_keys=True)
        else:
            with open(self.path, 'w') as f:
                self.json = {'fold':self.fold, 'epoch':0}
                f.truncate(0)
                json.dump(self.json, f, indent=4, sort_keys=True)

    def on_epoch_end(self, epoch: int, **kwargs):
        self.json['epoch'] = epoch
        with open(self.path, 'r+') as f:
            if self.truncate:
                f.truncate(0)
            json.dump(self.json, f, indent=4, sort_keys=True)

def compute_prediction_utility(labels, predictions, dt_early=-12, dt_optimal=-6, dt_late=3.0, max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0, check_errors=True):
    max_u_tp = 1
    min_u_fn = -2
    u_fp     = -0.05
    u_tn     = 0
    # Check inputs for errors.
    if check_errors:
        if len(predictions) != len(labels):
            raise Exception('Numbers of predictions and labels must be the same.')

        for label in labels:
            if not label in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')

        for prediction in predictions:
            if not prediction in (0, 1):
                raise Exception('Predictions must satisfy prediction == 0 or prediction == 1.')

        if dt_early >= dt_optimal:
            raise Exception('The earliest beneficial time for predictions must be before the optimal time.')

        if dt_optimal >= dt_late:
            raise Exception('The optimal time for predictions must be before the latest beneficial time.')

    # Does the patient eventually have sepsis?
    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels) - dt_optimal
    else:
        is_septic = False
        t_sepsis = float('inf')

    n = len(labels)

    # Define slopes and intercept points for utility functions of the form
    # u = m * t + b.
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    # Compare predicted and true conditions.
    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            # TP
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            # FP
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            # FN
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            # TN
            elif not is_septic and not predictions[t]:
                u[t] = u_tn

    # Find total utility for patient.
    return np.sum(u)

def compute_accuracy_f_measure(labels, predictions, check_errors=True):
    # Check inputs for errors.
    if check_errors:
        if len(predictions) != len(labels):
            raise Exception('Numbers of predictions and labels must be the same.')

        for label in labels:
            if not label in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')

        for prediction in predictions:
            if not prediction in (0, 1):
                raise Exception('Predictions must satisfy prediction == 0 or prediction == 1.')

    # Populate contingency table.
    n = len(labels)
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in range(n):
        if labels[i] and predictions[i]:
            tp += 1
        elif not labels[i] and predictions[i]:
            fp += 1
        elif labels[i] and not predictions[i]:
            fn += 1
        elif not labels[i] and not predictions[i]:
            tn += 1

    # Summarize contingency table.
    if tp + fp + fn + tn:
        accuracy = float(tp + tn) / float(tp + fp + fn + tn)
    else:
        accuracy = 1.0

    if 2 * tp + fp + fn:
        f_measure = float(2 * tp) / float(2 * tp + fp + fn)
    else:
        f_measure = 1.0

    return accuracy, f_measure

def compute_auc(labels, predictions, check_errors=True):
    # Check inputs for errors.
    if check_errors:
        if len(predictions) != len(labels):
            raise Exception('Numbers of predictions and labels must be the same.')

        for label in labels:
            if not label in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')

        for prediction in predictions:
            if not 0 <= prediction <= 1:
                warnings.warn('Predictions do not satisfy 0 <= prediction <= 1.')

    # Find prediction thresholds.
    thresholds = np.unique(predictions)[::-1]
    if thresholds[0] != 1:
        thresholds = np.insert(thresholds, 0, 1)
    if thresholds[-1] == 0:
        thresholds = thresholds[:-1]

    n = len(labels)
    m = len(thresholds)

    # Populate contingency table across prediction thresholds.
    tp = np.zeros(m)
    fp = np.zeros(m)
    fn = np.zeros(m)
    tn = np.zeros(m)

    # Find indices that sort the predicted probabilities from largest to
    # smallest.
    idx = np.argsort(predictions)[::-1]

    i = 0
    for j in range(m):
        # Initialize contingency table for j-th prediction threshold.
        if j == 0:
            tp[j] = 0
            fp[j] = 0
            fn[j] = np.sum(labels)
            tn[j] = n - fn[j]
        else:
            tp[j] = tp[j - 1]
            fp[j] = fp[j - 1]
            fn[j] = fn[j - 1]
            tn[j] = tn[j - 1]

        # Update contingency table for i-th largest predicted probability.
        while i < n and predictions[idx[i]] >= thresholds[j]:
            if labels[idx[i]]:
                tp[j] += 1
                fn[j] -= 1
            else:
                fp[j] += 1
                tn[j] -= 1
            i += 1

    # Summarize contingency table.
    tpr = np.zeros(m)
    tnr = np.zeros(m)
    ppv = np.zeros(m)
    npv = np.zeros(m)

    for j in range(m):
        if tp[j] + fn[j]:
            tpr[j] = tp[j] / (tp[j] + fn[j])
        else:
            tpr[j] = 1
        if fp[j] + tn[j]:
            tnr[j] = tn[j] / (fp[j] + tn[j])
        else:
            tnr[j] = 1
        if tp[j] + fp[j]:
            ppv[j] = tp[j] / (tp[j] + fp[j])
        else:
            ppv[j] = 1
        if fn[j] + tn[j]:
            npv[j] = tn[j] / (fn[j] + tn[j])
        else:
            npv[j] = 1

    # Compute AUROC as the area under a piecewise linear function with TPR /
    # sensitivity (x-axis) and TNR / specificity (y-axis) and AUPRC as the area
    # under a piecewise constant with TPR / recall (x-axis) and PPV / precision
    # (y-axis).
    auroc = 0
    auprc = 0
    for j in range(m-1):
        auroc += 0.5 * (tpr[j + 1] - tpr[j]) * (tnr[j + 1] + tnr[j])
        auprc += (tpr[j + 1] - tpr[j]) * ppv[j + 1]

    return auroc, auprc

def normalised_utility(y, preds, *args, **kwargs):
    num_rows          = len(y)
    observed_predictions = preds
    best_predictions     = np.zeros(num_rows)
    worst_predictions    = np.zeros(num_rows)
    inaction_predictions = np.zeros(num_rows)

    if np.any(y):
        t_sepsis = np.argmax(y) - dt_optimal
        best_predictions[max(0, t_sepsis + dt_early) : min(t_sepsis + dt_late + 1, num_rows)] = 1
    worst_predictions = 1 - best_predictions
    
    observed_utility = compute_prediction_utility(y, observed_predictions, **kwargs)
    best_utility     = compute_prediction_utility(y, best_predictions, **kwargs)
    worst_utility    = compute_prediction_utility(y, worst_predictions, **kwargs)
    inaction_utility = compute_prediction_utility(y, inaction_predictions, **kwargs)
    normalized_observed_utility = (observed_utility - inaction_utility) \
    / (best_utility - inaction_utility) if best_utility > 0 else 0
    return normalized_observed_utility

def get_normalised_utility_score(cohort_labels, cohort_predictions, cohort_probabilities,
                                 dt_early=-12, dt_optimal=-6, dt_late=3.0, max_u_tp=1,
                                 min_u_fn=-2, u_fp=-0.05, u_tn=0):
    
    num_files     = len(cohort_labels)

    labels        = np.concatenate(cohort_labels)
    predictions   = np.concatenate(cohort_predictions)
    probabilities = np.concatenate(cohort_probabilities)

    auroc, auprc        = compute_auc(labels, probabilities)
    accuracy, f_measure = compute_accuracy_f_measure(labels, predictions)
    
    # Compute utility.
    observed_utilities = np.zeros(num_files)
    best_utilities     = np.zeros(num_files)
    worst_utilities    = np.zeros(num_files)
    inaction_utilities = np.zeros(num_files)
    
    for k in range(num_files):
        labels = cohort_labels[k]
        num_rows          = len(labels)
        observed_predictions = cohort_predictions[k]
        best_predictions     = np.zeros(num_rows)
        worst_predictions    = np.zeros(num_rows)
        inaction_predictions = np.zeros(num_rows)

        if np.any(labels):
            t_sepsis = np.argmax(labels) - dt_optimal
            best_predictions[int(max(0, t_sepsis + dt_early)) : int(min(t_sepsis + dt_late + 1, num_rows))] = 1
        worst_predictions = 1 - best_predictions

        observed_utilities[k] = compute_prediction_utility(labels, observed_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        best_utilities[k]     = compute_prediction_utility(labels, best_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        worst_utilities[k]    = compute_prediction_utility(labels, worst_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        inaction_utilities[k] = compute_prediction_utility(labels, inaction_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)

    unnormalized_observed_utility = np.sum(observed_utilities)
    unnormalized_best_utility     = np.sum(best_utilities)
    unnormalized_worst_utility    = np.sum(worst_utilities)
    unnormalized_inaction_utility = np.sum(inaction_utilities)

    normalized_observed_utility = (unnormalized_observed_utility - unnormalized_inaction_utility) / (unnormalized_best_utility - unnormalized_inaction_utility)
    return auroc, auprc, accuracy, f_measure, normalized_observed_utility