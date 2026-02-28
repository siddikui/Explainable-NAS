import math
import torch
import torch.nn as nn
import time
import numpy as np
import random
import logging, sys, os
from torch.nn import Module
from torch.nn import Dropout
from torch.nn import MaxPool2d
from torch.nn import Sequential
from torch.nn import Conv2d, Linear
from torch.nn import BatchNorm1d, BatchNorm2d
from torch.nn import ReLU, Sigmoid
from torch.nn import PReLU

# === MODEL ANALYSIS ===================================================================================================
def general_num_params(model):
    # return number of differential parameters of input model
    return sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])

# keep a counter of available time
class Clock:
    def __init__(self, time_available):
        self.start_time =  time.time()
        self.total_time = time_available

    def check(self):
        return self.total_time + self.start_time - time.time()


# use this however you need\

# given a number of seconds, these two functions combine to print it out in a human-readable format
# I use these to print out status updates during my training loop
def div_remainder(n, interval):
    # finds divisor and remainder given some n/interval
    factor = math.floor(n / interval)
    remainder = int(n - (factor * interval))
    return factor, remainder


def show_time(seconds):
    # show amount of time as human readable
    if seconds < 60:
        return "{:.2f}s".format(seconds)
    elif seconds < (60 * 60):
        minutes, seconds = div_remainder(seconds, 60)
        return "{}m,{}s".format(minutes, seconds)
    else:
        hours, seconds = div_remainder(seconds, 60 * 60)
        minutes, seconds = div_remainder(seconds, 60)
        return "{}h,{}m,{}s".format(hours, minutes, seconds)

def red_blcks(input_shape):
    count = 0
    
    lesser = min(input_shape[2], input_shape[3])
    
    if input_shape[2] > 56 or input_shape[3] > 56:
        min_dim_red = 16
    else:
        min_dim_red = 11
    
    while(lesser > 11):
        
        lesser /= 2
        # print(lesser)
        count+=1
    #print('Reduction Blocks: ', count) 
    return count


class BackboneSearchClassifier(nn.Module):
    def __init__(self, units, in_c, num_classes,
                 input_channels, num_reductions):
        super().__init__()

        self.backbone = BackboneSearch(
            units=units,
            in_c=in_c,
            input_channels=input_channels,
            num_reductions=num_reductions
        )

        # --- dynamically infer feature size ---
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(2, input_channels, 28, 28)  # batch=2
            feat, _ = self.backbone(dummy)
            out_dim = feat.shape[1]
        self.backbone.train()


        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        feat, _ = self.backbone(x)
        logits = self.classifier(feat)
        return logits


class NetworkMix(nn.Module):
    def __init__(self, channels, metadata, layers, arch_ops, arch_kernel):
        super().__init__()

        input_channels = metadata["input_shape"][1]
        num_classes = metadata["num_classes"]

        # Depth search (NOT channels!)
        units = {
            'u1': 2,
            'u2': 2,
            'u3': 2,
            'u4': 2
        }
        num_reductions = red_blcks(metadata["input_shape"])
        self.model = BackboneSearchClassifier(
            units=units,
            in_c=channels,
            num_classes=num_classes,
            input_channels=input_channels,
            num_reductions=num_reductions
        )

    def forward(self, x):
        return self.model(x)


class BackboneSearch(nn.Module):
    def __init__(self, units, in_c, input_channels, num_reductions):
        super().__init__()

        self.units = units
        self.in_c = in_c

        print("BackboneSearch: in_c = ", in_c)
        print("BackboneSearch: num_reductions = ", num_reductions)
        print("BackboneSearch: units = ", units)
        print("BackboneSearch: input_channels = ", input_channels)

        # -------- INPUT ----------
        self.input_layer = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.PReLU(input_channels)
        )

        # -------- BODY ----------
        stage_names = ['u1', 'u2', 'u3', 'u4']
        downsample_stages = stage_names[:num_reductions]

        modules = []
        current_channels = input_channels
        reduction_count = 0

        print("\nStage | Channels")
        print("------|---------")
        print("Input |", current_channels)
        for stage in stage_names:
            for i in range(units[stage]):

                if stage in downsample_stages and i == 0:
                    stride = 2

                    # FIRST reduction: jump to in_c
                    if reduction_count == 0:
                        out_channels = in_c
                    else:
                        out_channels = current_channels * 2

                    reduction_count += 1
                else:
                    stride = 1
                    out_channels = current_channels

                modules.append(
                    BasicBlockIR(
                        current_channels,
                        out_channels,
                        stride
                    )
                )

                current_channels = out_channels


            print(stage, " |", current_channels)

        self.body = nn.Sequential(*modules)

        final_output_channel = current_channels
        print("\nFinal output channels:", final_output_channel)

        # -------- OUTPUT ----------
        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(final_output_channel),
            nn.Dropout(0.4),
            Flatten(),
            nn.Linear(final_output_channel, final_output_channel),
            nn.BatchNorm1d(final_output_channel, affine=False)
        )

        initialize_weights(self.modules())

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)

        norm = torch.norm(x, 2, 1, True)
        output = torch.div(x, norm)
        return output, norm

def initialize_weights(modules):
    """ Weight initilize, conv2d and linear is initialized with kaiming_normal
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()

                
class Flatten(Module):
    """ Flat tensor
    """
    def forward(self, input):
        return input.view(input.size(0), -1)

class BasicBlockIR(Module):
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIR, self).__init__()

        # shortcut
        if stride == 2 or in_channel != depth:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth)
            )
        else:
            self.shortcut_layer = nn.Identity()

        # RES: channel change ONLY happens on strided conv
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, in_channel, 3, 1, 1, bias=False),  # keep channels
            BatchNorm2d(in_channel),
            PReLU(in_channel),

            Conv2d(in_channel, depth, 3, stride, 1, bias=False), # change channels here
            BatchNorm2d(depth)
        )
    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log_lines(n):
    for i in range(n):
        logging.info('#############################################################################')

def get_per_dataset_time_limits(total_time, dataset_idx, n_datasets, search_ratio=0.6, train_ratio=0.4, extra_ratio=0.0125):
    """
    Given total_time (seconds), dataset index (0-based), and number of datasets,
    reserves extra_ratio (default 3%) for extras, and divides the rest equally among datasets.
    Returns (search_time, train_time, extra_time) for the current dataset.
    """
    if n_datasets < 1:
        raise ValueError("n_datasets must be >= 1")
    if total_time < 5400:
        extra_time = 45#total_time * extra_ratio * 2
    else:
        extra_time = 900#total_time * extra_ratio
    usable_time = total_time - extra_time
    per_dataset_time = usable_time / n_datasets
    search_time = per_dataset_time * search_ratio
    train_time = per_dataset_time * train_ratio
    per_data_extra = extra_time / n_datasets
    
    return search_time, train_time, per_data_extra

