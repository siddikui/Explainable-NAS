import math
import torch
import torch.nn as nn
import time
import numpy as np
import random
import logging, sys, os
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

class Conv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(Conv, self).__init__()
    self.op = nn.Sequential(

      nn.ReLU(inplace=False),
      nn.Dropout(p=0.2),
      nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),      
      nn.BatchNorm2d(C_out, affine=affine)
      )

  def forward(self, x):
    return self.op(x)

class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Dropout(p=0.2),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class NetworkMix(nn.Module):

  def __init__(self, C, metadata, layers, mixnet_code, k_size):
    start_layer = metadata['input_shape'][1]
    num_classes = metadata['num_classes']
    super(NetworkMix, self).__init__()
    self._layers = layers
    # input shape if min 48 then stem_multiplier is  0.25
    # take min out of 3 and 4
    #if metadata['input_shape'][2] > 64 or metadata['input_shape'][3] > 64:
    #  stem_multiplier = 0.0625
    if metadata['input_shape'][2] > 16 or metadata['input_shape'][3] > 16:
      stem_multiplier = 0.25
    else:
      stem_multiplier = 0.50

    C_curr = int(stem_multiplier*C)
    #C_curr = int(stem_multiplier*start_layer)
    self.stem = nn.Sequential(
      nn.Conv2d(start_layer, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    

    C_prev, C_curr = C_curr, C_curr
    
    self.mixlayers = nn.ModuleList()
    # reduction_prev = False
    '''Number of reduction blocks return from the block calculation function
    Genrate the list of of the reduction layers by using the totall nmber of layer
    '''
    num_reduction_blocks = red_blcks(metadata['input_shape'])
    reduction_list = [i * layers // (num_reduction_blocks+1) for i in range(1, num_reduction_blocks+1)]
    # Channel doubling logic: at each reduction block, channels double (C, 2C, 4C, ...)
    channel_multiplier = 1
    for i in range(layers):
      if i in reduction_list:
      
        
        if i != reduction_list[0]:
          channel_multiplier *= 2  # Double channels at each reduction block
        else:
          channel_multiplier = 1
        '''  
        
        channel_multiplier *= 2
        '''
        C_curr = C * channel_multiplier
        
        reduction = True
      else:
        reduction = False
      stride = 2 if reduction else 1
      if k_size[i]==3:
        pad=1
      elif k_size[i]==5: 
        pad=2
      else:
        pad=3
      if mixnet_code[i] == 0:
        mixlayer = SepConv(C_prev, C_curr, kernel_size=k_size[i], stride=stride, padding=pad, affine=True)
      else:
        mixlayer = Conv(C_prev, C_curr, kernel_size=k_size[i], stride=stride, padding=pad, affine=True)
      self.mixlayers += [mixlayer]
      C_prev = C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, x):
    # print("Input shape:", x.shape)

    x = self.stem(x)
    # print("Stem output shape:", x.shape)

    for i, mixlayer in enumerate(self.mixlayers):
        x = mixlayer(x)
        # print(f"MixLayer {i+1} output shape:", x.shape)

    out = self.global_pooling(x)
    # print("Global pooling output shape:", out.shape)
    logits = self.classifier(out.view(out.size(0), -1))
    # print("Logits output shape:", logits.shape)
    return logits


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

