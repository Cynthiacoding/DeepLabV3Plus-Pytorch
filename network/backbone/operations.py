import torch
import torch.nn as nn

OPS = {
    'mbconv_k3_t1': lambda C_in, C_out, stride, affine, track_running_stats: MBConv(C_in, C_out, 3, stride, 1, t=1, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k3_t3': lambda C_in, C_out, stride, affine, track_running_stats: MBConv(C_in, C_out, 3, stride, 1, t=3, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k3_t6': lambda C_in, C_out, stride, affine, track_running_stats: MBConv(C_in, C_out, 3, stride, 1, t=6, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k5_t3': lambda C_in, C_out, stride, affine, track_running_stats: MBConv(C_in, C_out, 5, stride, 2, t=3, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k5_t6': lambda C_in, C_out, stride, affine, track_running_stats: MBConv(C_in, C_out, 5, stride, 2, t=6, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k7_t3': lambda C_in, C_out, stride, affine, track_running_stats: MBConv(C_in, C_out, 7, stride, 3, t=3, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k7_t6': lambda C_in, C_out, stride, affine, track_running_stats: MBConv(C_in, C_out, 7, stride, 3, t=6, affine=affine, track_running_stats=track_running_stats),
    'basic_block': lambda C_in, C_out, stride, affine, track_running_stats: BasicBlock(C_in, C_out, stride, affine=affine, track_running_stats=track_running_stats),
    'bottle_neck': lambda C_in, C_out, stride, affine, track_running_stats: Bottleneck(C_in, C_out, stride, affine=affine, track_running_stats=track_running_stats),
    'skip_connect': lambda C_in, C_out, stride, affine, track_running_stats: Skip(C_in, C_out, 1, affine=affine, track_running_stats=track_running_stats),
}

PRIMITIVES_stack= ['mbconv_k3_t3',
                    'mbconv_k3_t6',
                    'mbconv_k5_t3',
                    'mbconv_k5_t6',
                    'mbconv_k7_t3',
                    'mbconv_k7_t6',
                    'skip_connect',]
PRIMITIVES_head= ['mbconv_k3_t3',
                    'mbconv_k3_t6',
                    'mbconv_k5_t3',
                    'mbconv_k5_t6',
                    'mbconv_k7_t3',
                    'mbconv_k7_t6',
                    ]


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


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 affine=True, track_running_stats=True):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, affine=affine, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, affine=affine, track_running_stats=track_running_stats)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes, affine=affine, track_running_stats=track_running_stats),
            )

    def forward(self, x):  
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, affine=True, track_running_stats=True):
        super(Bottleneck, self).__init__()
        if inplanes != 32:
            inplanes *= 4
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample = None
        if stride != 1 or inplanes != planes*4:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * 4, stride),
                nn.BatchNorm2d(planes * 4, affine=affine, track_running_stats=track_running_stats),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


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

class Identity(nn.Module):
    def __init__(self, stride):
        super(Identity, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x
        else:
            return x[:, :, ::self.stride, ::self.stride]



CANDIDATE_BLOCKS = ['mbconv_k3_t3',
                    'mbconv_k3_t6',
                    'skip_connect']

# **** to recalculate latency use command:
# l_table = LookUpTable(calulate_latency=True, path_to_file='lookup_table.txt', cnt_of_runs=50)
# results will be written to './supernet_functions/lookup_table.txt''
# **** to read latency from the another file use command:
# l_table = LookUpTable(calulate_latency=False, path_to_file='lookup_table.txt')
class LookUpTable:
    def __init__(self, candidate_blocks=CANDIDATE_BLOCKS, search_space=SEARCH_SPACE,
                 calulate_latency=False):
        self.cnt_layers = len(search_space["input_shape"])
        # constructors for each operation
        self.lookup_table_operations = {op_name : PRIMITIVES[op_name] for op_name in candidate_blocks}
        # arguments for the ops constructors. one set of arguments for all 9 constructors at each layer
        # input_shapes just for convinience
        self.layers_parameters, self.layers_input_shapes = self._generate_layers_parameters(search_space)
        
        # lookup_table
        self.lookup_table_latency = None
        if calulate_latency:
            self._create_from_operations(cnt_of_runs=CONFIG_SUPERNET['lookup_table']['number_of_runs'],
                                         write_to_file=CONFIG_SUPERNET['lookup_table']['path_to_lookup_table'])
        else:
            self._create_from_file(path_to_file=CONFIG_SUPERNET['lookup_table']['path_to_lookup_table'])
    
    def _generate_layers_parameters(self, search_space):
        # layers_parameters are : C_in, C_out, expansion, stride
        layers_parameters = [(search_space["input_shape"][layer_id][0],
                              search_space["channel_size"][layer_id],
                              # expansion (set to -999) embedded into operation and will not be considered
                              # (look fbnet_building_blocks/fbnet_builder.py - this is facebookresearch code
                              # and I don't want to modify it)
                              -999,
                              search_space["strides"][layer_id]
                             ) for layer_id in range(self.cnt_layers)]
        
        # layers_input_shapes are (C_in, input_w, input_h)
        layers_input_shapes = search_space["input_shape"]
        
        return layers_parameters, layers_input_shapes
    
    # CNT_OP_RUNS us number of times to check latency (we will take average)
    def _create_from_operations(self, cnt_of_runs, write_to_file=None):
        self.lookup_table_latency = self._calculate_latency(self.lookup_table_operations,
                                                            self.layers_parameters,
                                                            self.layers_input_shapes,
                                                            cnt_of_runs)
        if write_to_file is not None:
            self._write_lookup_table_to_file(write_to_file)
    
    def _calculate_latency(self, operations, layers_parameters, layers_input_shapes, cnt_of_runs):
        LATENCY_BATCH_SIZE = 1
        latency_table_layer_by_ops = [{} for i in range(self.cnt_layers)]
        
        for layer_id in range(self.cnt_layers):
            for op_name in operations:
                op = operations[op_name](*layers_parameters[layer_id])
                input_sample = torch.randn((LATENCY_BATCH_SIZE, *layers_input_shapes[layer_id]))
                globals()['op'], globals()['input_sample'] = op, input_sample
                total_time = timeit.timeit('output = op(input_sample)', setup="gc.enable()", \
                                           globals=globals(), number=cnt_of_runs)
                # measured in micro-second
                latency_table_layer_by_ops[layer_id][op_name] = total_time / cnt_of_runs / LATENCY_BATCH_SIZE * 1e6
                
        return latency_table_layer_by_ops
    
    def _write_lookup_table_to_file(self, path_to_file):
        clear_files_in_the_list([path_to_file])
        ops = [op_name for op_name in self.lookup_table_operations]
        text = [op_name + " " for op_name in ops[:-1]]
        text.append(ops[-1] + "\n")
        
        for layer_id in range(self.cnt_layers):
            for op_name in ops:
                text.append(str(self.lookup_table_latency[layer_id][op_name]))
                text.append(" ")
            text[-1] = "\n"
        text = text[:-1]
        
        text = ''.join(text)
        add_text_to_file(text, path_to_file)
    
    def _create_from_file(self, path_to_file):
        self.lookup_table_latency = self._read_lookup_table_from_file(path_to_file)
    
    def _read_lookup_table_from_file(self, path_to_file):
        latences = [line.strip('\n') for line in open(path_to_file)]
        ops_names = latences[0].split(" ")
        latences = [list(map(float, layer.split(" "))) for layer in latences[1:]]
        
        lookup_table_latency = [{op_name : latences[i][op_id] 
                                      for op_id, op_name in enumerate(ops_names)
                                     } for i in range(self.cnt_layers)]
        return lookup_table_latency
