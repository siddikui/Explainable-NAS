import math
import torch
import torch.nn as nn
import time
import numpy as np
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
    while(lesser > 4):
        lesser /= 2
        # print(lesser)
        count+=1
    print('Reduction Blocks: ', count) 
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
    
    stem_multiplier = 0.25
    C_curr = int(stem_multiplier*C)
    self.stem = nn.Sequential(
      nn.Conv2d(start_layer, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    
    C_prev, C_curr = C_curr, C
    
    self.mixlayers = nn.ModuleList()
    reduction_prev = False
    '''Number of reduction blocks return from the block calculation function
    Genrate the list of of the reduction layers by using the totall nmber of layer
    '''
    num_reduction_blocks = red_blcks(metadata['input_shape'])
    reduction_list = [i * layers // (num_reduction_blocks+1) for i in range(1, num_reduction_blocks+1)]
    
    for i in range(layers):
      if i in reduction_list:
        C_curr = C * (reduction_list.index(i)+2)
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

      reduction_prev = reduction
        
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

