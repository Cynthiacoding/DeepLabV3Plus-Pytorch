import os
import numpy as np
from pdb import set_trace as bp
from operations import *

H = 1024
W = 2048
scale_range = [8, 16, 32]
widths_range = [4./12, 6./12, 8./12, 10./12, 1.]
file_name = "latency_lookup_table.npy"
if os.path.isfile(file_name):
    lookup_table = np.load(file_name).item()
else:
    lookup_table = {}