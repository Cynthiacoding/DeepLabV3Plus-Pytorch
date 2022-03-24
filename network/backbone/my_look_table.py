import numpy as np
import torch
from collections import OrderedDict
import torch.nn as nn
import time
OPS = {
    'mbconv_k3_t1': lambda C_in, C_out, stride, affine, track_running_stats: MBConv(C_in, C_out, 3, stride, 1, t=1, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k3_t6': lambda C_in, C_out, stride, affine, track_running_stats: MBConv(C_in, C_out, 3, stride, 1, t=6, affine=affine, track_running_stats=track_running_stats),
    'skip_connect': lambda C_in, C_out, stride, affine, track_running_stats: Skip(C_in, C_out, 1, affine=affine, track_running_stats=track_running_stats),
}


class MBConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, t=3, affine=True, 
                    track_running_stats=True, use_se=False):
        super(MBConv, self).__init__()
        self.t = t
        if self.t > 1:
            self._expand_conv = nn.Sequential(
                nn.Conv2d(C_in, C_in*self.t, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(C_in*self.t, affine=affine, track_running_stats=track_running_stats),
                nn.ReLU6(inplace=True))

            self._depthwise_conv = nn.Sequential(
                nn.Conv2d(C_in*self.t, C_in*self.t, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in*self.t, bias=False),
                nn.BatchNorm2d(C_in*self.t, affine=affine, track_running_stats=track_running_stats),
                nn.ReLU6(inplace=True))

            self._project_conv = nn.Sequential(
                nn.Conv2d(C_in*self.t, C_out, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats))
        else:
            self._expand_conv = None

            self._depthwise_conv = nn.Sequential(
                nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
                nn.BatchNorm2d(C_in, affine=affine, track_running_stats=track_running_stats),
                nn.ReLU6(inplace=True))

            self._project_conv = nn.Sequential(
                nn.Conv2d(C_in, C_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(C_out))

    def forward(self, x):
        input_data = x
        if self._expand_conv is not None:
            x = self._expand_conv(x)
        x = self._depthwise_conv(x)
        out_data = self._project_conv(x)

        if out_data.shape == input_data.shape:
            return out_data + input_data
        else:
            return out_data


class Identity(nn.Module):
    def __init__(self, stride):
        super(Identity, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x
        else:
            return x[:, :, ::self.stride, ::self.stride]

class Skip(nn.Module):
    def __init__(self, C_in, C_out, stride, affine=True, track_running_stats=True):
        super(Skip, self).__init__()
        if C_in!=C_out:
            skip_conv = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size=1, stride=stride, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats))
            stride = 1
        self.op=Identity(stride)

        if C_in!=C_out:
            self.op=nn.Sequential(skip_conv, self.op)

    def forward(self,x):
        return self.op(x)


CANDIDATE_BLOCKS = ['mbconv_k3_t6',
                    'skip_connect']

#### table 1. input shapes of 16 searched layers (considering with strides)
    # Note: the second and third dimentions are recommended (will not be used in training) and written just for debagging
SEARCH_SPACE = OrderedDict([
    ("input_shape", [(24, 256, 256), #2
                     (32, 128, 128), #3
                     (32, 128, 128), #4
                     (48, 64, 64),#5
                     (48, 64, 64),#6
                     (48, 32, 32),#7
                     (88, 16, 16),#8
                     (88, 16, 16),#9
                     (88, 16, 16),#10 
                     (88, 16, 16),#11
                     (136, 16, 16),#12
                     (136, 16, 16),#13
                     (136, 16, 16),#14
                     (224, 16, 16),#15
                     (224, 16, 16),#16
                     (224, 16, 16),#17
                     ]),
    ("channel_size", [32, 32, 48, 48, 48, 88, 88, 88, 88, 136, 136, 136, 224, 224, 224, 448]),
    ("strides", [2,1,
                 2,1,1,
                 2,1,1,1,1,1,1,1,1,1,1])                     
                     
])                     
                     
                     
class LookUpTable:
    def __init__(self, candidate_blocks=CANDIDATE_BLOCKS, search_space=SEARCH_SPACE,
                 calulate_latency=False):
        self.cnt_layers = len(search_space["input_shape"])
        # constructors for each operation
        self.lookup_table_operations = {op_name : OPS[op_name] for op_name in candidate_blocks}
        # arguments for the ops constructors. one set of arguments for all 9 constructors at each layer
        # input_shapes just for convinience
        self.layers_parameters, self.layers_input_shapes = self._generate_layers_parameters(search_space)

    def _generate_layers_parameters(self, search_space):
        # layers_parameters are : C_in, C_out, stride, affine, track_running_stats
        layers_parameters = [(search_space["input_shape"][layer_id][0],
                              search_space["channel_size"][layer_id],
                              search_space["strides"][layer_id],
                              True,
                              True) for layer_id in range(self.cnt_layers)]
        
        # layers_input_shapes are (C_in, input_w, input_h)
        layers_input_shapes = search_space["input_shape"]                     
        
        return layers_parameters, layers_input_shapes
                     


class CollectData(nn.Module):
    def __init__(self, input_channel=24, output_channel=36, stride=1):
        super(CollectData, self).__init__()
        block = MBConv
        times = 30
        self.features = []
        for i in range(times):
            self.features.append(block(input_channel, output_channel, 3, stride, 1, t=6))

    def forward(self, x):
        torch.cuda.synchronize()
        start_time = time.time()
        for each_block in self.features:
            res = each_block(x)
        torch.cuda.synchronize()
        end_time = time.time()

        return end_time-start_time 

def collect(SEARCH_SPACE, load_time, log_txt=None):

    search_space = SEARCH_SPACE
    for layer_id in range(16):
        input_channel = search_space["input_shape"][layer_id][0]
        input_shape = search_space["input_shape"][layer_id][1]
        output_channel = search_space["channel_size"][layer_id]
        stride = search_space["strides"][layer_id]
        collectData = CollectData(input_channel, output_channel, stride)
        image = torch.rand(1, input_channel, input_shape, input_shape)
        collect_time = collectData(image)
        load_time[layer_id]+=collect_time
        if stride==2:
            output_shape = int(input_shape/2)
        else:
            output_shape = input_shape
        property_cov = "ConvBN-input:"+ str(input_shape)+ "x"+ str(input_shape)+ "x" +str(input_channel)+"-output:"+str(output_shape)+ "x"+ str(output_shape)+ "x" +str(output_channel)+"-stride:"+str(stride)
        if log_txt!=None:
            with open(log_txt,"a") as f:
                f.write(property_cov+ ":"+ str(collect_time)+ "\n")

log_txt = "/data/renhongzhang/DeepLabV3Plus-Pytorch/network/backbone/log.txt"
load_time = torch.zeros(16)
collect(SEARCH_SPACE,load_time)
print(load_time)
collect(SEARCH_SPACE,load_time)
print(load_time)
collect(SEARCH_SPACE,load_time)
print(load_time)
collect(SEARCH_SPACE,load_time)
collect(SEARCH_SPACE,load_time)





