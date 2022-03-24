import numpy as np
import torch
from collections import OrderedDict
import torch.nn as nn
from thop import profile
import os.path as osp

try:
    from utils import compute_latency_ms_tensorrt as compute_latency
    print("use TensorRT for latency test")
except:
    from utils import compute_latency_ms_pytorch as compute_latency
    print("use PyTorch for latency test") 
latency_lookup_table = {}
table_file_name = "latency_lookup_table.npy"
if osp.isfile(table_file_name):
    latency_lookup_table = np.load(table_file_name).item()


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

    # @staticmethod
    # def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
    #     layer = MBConv(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
    #     flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), verbose=False)
    #     return flops
    
    # @staticmethod
    # def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
    #     layer = MBConv(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
    #     latency = compute_latency(layer, (1, C_in, h, w))
    #     return latency

    # def forward_latency(self, size):
    #     c_in, h_in, w_in = size
    #     if self.slimmable:
    #         assert c_in == int(self.C_in * self.ratio[0]), "c_in %d, self.C_in * self.ratio[0] %d"%(c_in, self.C_in * self.ratio[0])
    #         c_out = int(self.C_out * self.ratio[1])
    #     else:
    #         assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
    #         c_out = self.C_out
    #     if self.stride == 1:
    #         h_out = h_in; w_out = w_in
    #     else:
    #         h_out = h_in // 2; w_out = w_in // 2
    #     name = "ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d"%(h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
    #     if name in latency_lookup_table:
    #         latency = latency_lookup_table[name]
    #     else:
    #         print("not found in latency_lookup_table:", name)
    #         latency = MBConv._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
    #         latency_lookup_table[name] = latency
    #         np.save(table_file_name, latency_lookup_table)
    #     return latency, (c_out, h_out, w_out)


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