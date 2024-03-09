import math
import torch
import torch.nn as nn

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

  def __init__(self, C, num_classes, layers, mixnet_code, k_size, start_layer):
    super(NetworkMix, self).__init__()
    self._layers = layers
    
    # stem_multiplier = 2
    # C_curr = stem_multiplier*C
    C_curr = start_layer
    self.stem = nn.Sequential(
      nn.Conv2d(start_layer, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    
    C_prev, C_curr = C_curr, C
    
    self.mixlayers = nn.ModuleList()
    reduction_prev = False
    
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
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
    
    x = self.stem(x)
    
    for i, mixlayer in enumerate(self.mixlayers):
      x = mixlayer(x)
        
    out = self.global_pooling(x)
    logits = self.classifier(out.view(out.size(0),-1))
    
    return logits
