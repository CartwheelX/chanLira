# """
# MIT License

# Copyright (c) 2020-present TorchQuantum Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# """

# # from torchquantum.plugins.qiskit_plugin import tq2qiskit
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# import argparse
# import random
# import numpy as np

# import torchquantum as tq
# from torchquantum.plugin import (
#     tq2qiskit_measurement,
#     qiskit_assemble_circs,
#     op_history2qiskit,
#     op_history2qiskit_expand_params,
#     tq2qiskit
# )

# from torchquantum.dataset import MNIST
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from torchquantum.encoding.encodings import encoder_op_list_name_dict

# encoder_op_list_name_dict = {
#     "1x1_ry": [
#         {"input_idx": [0], "func": "ry", "wires": [0]},
#     ],
#     "2x1_ryry": [
#         {"input_idx": [0], "func": "ry", "wires": [0]},
#         {"input_idx": [1], "func": "ry", "wires": [1]},
#     ],
#     "2x8_rxryrzrxryrzrxry": [
#         {"input_idx": [0], "func": "rx", "wires": [0]},
#         {"input_idx": [1], "func": "rx", "wires": [1]},
#         {"input_idx": [2], "func": "ry", "wires": [0]},
#         {"input_idx": [3], "func": "ry", "wires": [1]},
#         {"input_idx": [4], "func": "rz", "wires": [0]},
#         {"input_idx": [5], "func": "rz", "wires": [1]},
#         {"input_idx": [6], "func": "rx", "wires": [0]},
#         {"input_idx": [7], "func": "rx", "wires": [1]},
#         {"input_idx": [8], "func": "ry", "wires": [0]},
#         {"input_idx": [9], "func": "ry", "wires": [1]},
#         {"input_idx": [10], "func": "rz", "wires": [0]},
#         {"input_idx": [11], "func": "rz", "wires": [1]},
#         {"input_idx": [12], "func": "rx", "wires": [0]},
#         {"input_idx": [13], "func": "rx", "wires": [1]},
#         {"input_idx": [14], "func": "ry", "wires": [0]},
#         {"input_idx": [15], "func": "ry", "wires": [1]},
#     ],
#     "3x1_ryryry": [
#         {"input_idx": [0], "func": "ry", "wires": [0]},
#         {"input_idx": [1], "func": "ry", "wires": [1]},
#         {"input_idx": [2], "func": "ry", "wires": [2]},
#     ],
#     "3x1_rxrxrx": [
#         {"input_idx": [0], "func": "rx", "wires": [0]},
#         {"input_idx": [1], "func": "rx", "wires": [1]},
#         {"input_idx": [2], "func": "rx", "wires": [2]},
#     ],
#     "4_ry": [
#         {"input_idx": [0], "func": "ry", "wires": [0]},
#         {"input_idx": [1], "func": "ry", "wires": [1]},
#         {"input_idx": [2], "func": "ry", "wires": [2]},
#         {"input_idx": [3], "func": "ry", "wires": [3]},
#     ],
#     "4x4_u3rx": [
#         {"input_idx": [0, 1, 2], "func": "u3", "wires": [0]},
#         {"input_idx": [3], "func": "rx", "wires": [0]},
#         {"input_idx": [4, 5, 6], "func": "u3", "wires": [1]},
#         {"input_idx": [7], "func": "rx", "wires": [1]},
#         {"input_idx": [8, 9, 10], "func": "u3", "wires": [2]},
#         {"input_idx": [11], "func": "rx", "wires": [2]},
#         {"input_idx": [12, 13, 14], "func": "u3", "wires": [3]},
#         {"input_idx": [15], "func": "rx", "wires": [3]},
#     ],
#     "4x4_u3_h_rx": [
#         {"input_idx": [0, 1, 2], "func": "u3", "wires": [0]},
#         {"input_idx": [3], "func": "rx", "wires": [0]},
#         {"func": "h", "wires": [0]},
#         {"func": "h", "wires": [1]},
#         {"func": "h", "wires": [2]},
#         {"func": "h", "wires": [3]},
#         {"input_idx": [4, 5, 6], "func": "u3", "wires": [1]},
#         {"input_idx": [7], "func": "rx", "wires": [1]},
#         {"input_idx": [8, 9, 10], "func": "u3", "wires": [2]},
#         {"input_idx": [11], "func": "rx", "wires": [2]},
#         {"input_idx": [12, 13, 14], "func": "u3", "wires": [3]},
#         {"input_idx": [15], "func": "rx", "wires": [3]},
#     ],
#     "4x4_ryzxy": [
#         {"input_idx": [0], "func": "ry", "wires": [0]},
#         {"input_idx": [1], "func": "ry", "wires": [1]},
#         {"input_idx": [2], "func": "ry", "wires": [2]},
#         {"input_idx": [3], "func": "ry", "wires": [3]},
#         {"input_idx": [4], "func": "rz", "wires": [0]},
#         {"input_idx": [5], "func": "rz", "wires": [1]},
#         {"input_idx": [6], "func": "rz", "wires": [2]},
#         {"input_idx": [7], "func": "rz", "wires": [3]},
#         {"input_idx": [8], "func": "rx", "wires": [0]},
#         {"input_idx": [9], "func": "rx", "wires": [1]},
#         {"input_idx": [10], "func": "rx", "wires": [2]},
#         {"input_idx": [11], "func": "rx", "wires": [3]},
#         {"input_idx": [12], "func": "ry", "wires": [0]},
#         {"input_idx": [13], "func": "ry", "wires": [1]},
#         {"input_idx": [14], "func": "ry", "wires": [2]},
#         {"input_idx": [15], "func": "ry", "wires": [3]},
#     ],

#     "4x4_staggered_axes": [
#         # // column 1
#         {"input_idx":[0],  "func":"ry","wires":[0]},
#         {"input_idx":[1],  "func":"rx","wires":[1]},
#         {"input_idx":[2],  "func":"rz","wires":[2]},
#         {"input_idx":[3],  "func":"ry","wires":[3]},
#         # // column 2
#         {"input_idx":[4],  "func":"rz","wires":[0]},
#         {"input_idx":[5],  "func":"ry","wires":[1]},
#         {"input_idx":[6],  "func":"rx","wires":[2]},
#         {"input_idx":[7],  "func":"rx","wires":[3]},
#         # // column 3
#         {"input_idx":[8],  "func":"rx","wires":[0]},
#         {"input_idx":[9],  "func":"rz","wires":[1]},
#         {"input_idx":[10], "func":"ry","wires":[2]},
#         {"input_idx":[11], "func":"rz","wires":[3]},
#         # // column 4
#         {"input_idx":[12], "func":"ry","wires":[0]},
#         {"input_idx":[13], "func":"rx","wires":[1]},
#         {"input_idx":[14], "func":"rz","wires":[2]},
#         {"input_idx":[15], "func":"ry","wires":[3]}
#         ],

#     "4x8_ryzxy_2pass": [
#         {"input_idx":[0],  "func":"ry","wires":[0]},
#         {"input_idx":[1],  "func":"ry","wires":[1]},
#         {"input_idx":[2],  "func":"ry","wires":[2]},
#         {"input_idx":[3],  "func":"ry","wires":[3]},
#         {"input_idx":[4],  "func":"rz","wires":[0]},
#         {"input_idx":[5],  "func":"rz","wires":[1]},
#         {"input_idx":[6],  "func":"rz","wires":[2]},
#         {"input_idx":[7],  "func":"rz","wires":[3]},
#         {"input_idx":[8],  "func":"rx","wires":[0]},
#         {"input_idx":[9],  "func":"rx","wires":[1]},
#         {"input_idx":[10], "func":"rx","wires":[2]},
#         {"input_idx":[11], "func":"rx","wires":[3]},
#         {"input_idx":[12], "func":"ry","wires":[0]},
#         {"input_idx":[13], "func":"ry","wires":[1]},
#         {"input_idx":[14], "func":"ry","wires":[2]},
#         {"input_idx":[15], "func":"ry","wires":[3]},

#         {"input_idx":[16], "func":"ry","wires":[0]},
#         {"input_idx":[17], "func":"ry","wires":[1]},
#         {"input_idx":[18], "func":"ry","wires":[2]},
#         {"input_idx":[19], "func":"ry","wires":[3]},
#         {"input_idx":[20], "func":"rz","wires":[0]},
#         {"input_idx":[21], "func":"rz","wires":[1]},
#         {"input_idx":[22], "func":"rz","wires":[2]},
#         {"input_idx":[23], "func":"rz","wires":[3]},
#         {"input_idx":[24], "func":"rx","wires":[0]},
#         {"input_idx":[25], "func":"rx","wires":[1]},
#         {"input_idx":[26], "func":"rx","wires":[2]},
#         {"input_idx":[27], "func":"rx","wires":[3]},
#         {"input_idx":[28], "func":"ry","wires":[0]},
#         {"input_idx":[29], "func":"ry","wires":[1]},
#         {"input_idx":[30], "func":"ry","wires":[2]},
#         {"input_idx":[31], "func":"ry","wires":[3]}
#         ],

#     "8x2_ry": [
#         {"input_idx": [0], "func": "ry", "wires": [0]},
#         {"input_idx": [1], "func": "ry", "wires": [1]},
#         {"input_idx": [2], "func": "ry", "wires": [2]},
#         {"input_idx": [3], "func": "ry", "wires": [3]},
#         {"input_idx": [4], "func": "ry", "wires": [4]},
#         {"input_idx": [5], "func": "ry", "wires": [5]},
#         {"input_idx": [6], "func": "ry", "wires": [6]},
#         {"input_idx": [7], "func": "ry", "wires": [7]},
#         {"input_idx": [8], "func": "ry", "wires": [0]},
#         {"input_idx": [9], "func": "ry", "wires": [1]},
#         {"input_idx": [10], "func": "ry", "wires": [2]},
#         {"input_idx": [11], "func": "ry", "wires": [3]},
#         {"input_idx": [12], "func": "ry", "wires": [4]},
#         {"input_idx": [13], "func": "ry", "wires": [5]},
#         {"input_idx": [14], "func": "ry", "wires": [6]},
#         {"input_idx": [15], "func": "ry", "wires": [7]},
#     ],
#     "16_ry": [
#         {"input_idx": [0], "func": "ry", "wires": [0]},
#         {"input_idx": [1], "func": "ry", "wires": [1]},
#         {"input_idx": [2], "func": "ry", "wires": [2]},
#         {"input_idx": [3], "func": "ry", "wires": [3]},
#         {"input_idx": [4], "func": "ry", "wires": [4]},
#         {"input_idx": [5], "func": "ry", "wires": [5]},
#         {"input_idx": [6], "func": "ry", "wires": [6]},
#         {"input_idx": [7], "func": "ry", "wires": [7]},
#         {"input_idx": [8], "func": "ry", "wires": [8]},
#         {"input_idx": [9], "func": "ry", "wires": [9]},
#         {"input_idx": [10], "func": "ry", "wires": [10]},
#         {"input_idx": [11], "func": "ry", "wires": [11]},
#         {"input_idx": [12], "func": "ry", "wires": [12]},
#         {"input_idx": [13], "func": "ry", "wires": [13]},
#         {"input_idx": [14], "func": "ry", "wires": [14]},
#         {"input_idx": [15], "func": "ry", "wires": [15]},
#     ],
#     "4x4_rzsx": [
#         {"input_idx": [0], "func": "rz", "wires": [0]},
#         {"input_idx": None, "func": "sx", "wires": [0]},
#         {"input_idx": [1], "func": "rz", "wires": [1]},
#         {"input_idx": None, "func": "sx", "wires": [1]},
#         {"input_idx": [2], "func": "rz", "wires": [2]},
#         {"input_idx": None, "func": "sx", "wires": [2]},
#         {"input_idx": [3], "func": "rz", "wires": [3]},
#         {"input_idx": None, "func": "sx", "wires": [3]},
#         {"input_idx": [4], "func": "rz", "wires": [0]},
#         {"input_idx": None, "func": "sx", "wires": [0]},
#         {"input_idx": [5], "func": "rz", "wires": [1]},
#         {"input_idx": None, "func": "sx", "wires": [1]},
#         {"input_idx": [6], "func": "rz", "wires": [2]},
#         {"input_idx": None, "func": "sx", "wires": [2]},
#         {"input_idx": [7], "func": "rz", "wires": [3]},
#         {"input_idx": None, "func": "sx", "wires": [3]},
#         {"input_idx": [8], "func": "rz", "wires": [0]},
#         {"input_idx": None, "func": "sx", "wires": [0]},
#         {"input_idx": [9], "func": "rz", "wires": [1]},
#         {"input_idx": None, "func": "sx", "wires": [1]},
#         {"input_idx": [10], "func": "rz", "wires": [2]},
#         {"input_idx": None, "func": "sx", "wires": [2]},
#         {"input_idx": [11], "func": "rz", "wires": [3]},
#         {"input_idx": None, "func": "sx", "wires": [3]},
#         {"input_idx": [12], "func": "rz", "wires": [0]},
#         {"input_idx": None, "func": "sx", "wires": [0]},
#         {"input_idx": [13], "func": "rz", "wires": [1]},
#         {"input_idx": None, "func": "sx", "wires": [1]},
#         {"input_idx": [14], "func": "rz", "wires": [2]},
#         {"input_idx": None, "func": "sx", "wires": [2]},
#         {"input_idx": [15], "func": "rz", "wires": [3]},
#         {"input_idx": None, "func": "sx", "wires": [3]},
#     ],
#     "15_ryrz": [
#         {"input_idx": [0], "func": "ry", "wires": [0]},
#         {"input_idx": [1], "func": "ry", "wires": [1]},
#         {"input_idx": [2], "func": "ry", "wires": [2]},
#         {"input_idx": [3], "func": "ry", "wires": [3]},
#         {"input_idx": [4], "func": "ry", "wires": [4]},
#         {"input_idx": [5], "func": "ry", "wires": [5]},
#         {"input_idx": [6], "func": "ry", "wires": [6]},
#         {"input_idx": [7], "func": "ry", "wires": [7]},
#         {"input_idx": [8], "func": "ry", "wires": [8]},
#         {"input_idx": [9], "func": "ry", "wires": [9]},
#         {"input_idx": [10], "func": "ry", "wires": [10]},
#         {"input_idx": [11], "func": "ry", "wires": [11]},
#         {"input_idx": [12], "func": "ry", "wires": [12]},
#         {"input_idx": [13], "func": "ry", "wires": [13]},
#         {"input_idx": [14], "func": "ry", "wires": [14]},
#         {"input_idx": [15], "func": "rz", "wires": [0]},
#     ],
#     "2x8_ryzxyzxyz": [
#         {"input_idx": [0], "func": "ry", "wires": [0]},
#         {"input_idx": [1], "func": "ry", "wires": [1]},
#         {"input_idx": [2], "func": "rz", "wires": [0]},
#         {"input_idx": [3], "func": "rz", "wires": [1]},
#         {"input_idx": [4], "func": "rx", "wires": [0]},
#         {"input_idx": [5], "func": "rx", "wires": [1]},
#         {"input_idx": [6], "func": "ry", "wires": [0]},
#         {"input_idx": [7], "func": "ry", "wires": [1]},
#         {"input_idx": [8], "func": "rz", "wires": [0]},
#         {"input_idx": [9], "func": "rz", "wires": [1]},
#         {"input_idx": [10], "func": "rx", "wires": [0]},
#         {"input_idx": [11], "func": "rx", "wires": [1]},
#         {"input_idx": [12], "func": "ry", "wires": [0]},
#         {"input_idx": [13], "func": "ry", "wires": [1]},
#         {"input_idx": [14], "func": "rz", "wires": [0]},
#         {"input_idx": [15], "func": "rz", "wires": [1]},
#     ],
#     "10_ryzx": [
#         {"input_idx": [0], "func": "ry", "wires": [0]},
#         {"input_idx": [1], "func": "ry", "wires": [1]},
#         {"input_idx": [2], "func": "ry", "wires": [2]},
#         {"input_idx": [3], "func": "ry", "wires": [3]},
#         {"input_idx": [4], "func": "rz", "wires": [0]},
#         {"input_idx": [5], "func": "rz", "wires": [1]},
#         {"input_idx": [6], "func": "rz", "wires": [2]},
#         {"input_idx": [7], "func": "rz", "wires": [3]},
#         {"input_idx": [8], "func": "rx", "wires": [0]},
#         {"input_idx": [9], "func": "rx", "wires": [1]},
#     ],
#     "10_ry": [
#         {"input_idx": [0], "func": "ry", "wires": [0]},
#         {"input_idx": [1], "func": "ry", "wires": [1]},
#         {"input_idx": [2], "func": "ry", "wires": [2]},
#         {"input_idx": [3], "func": "ry", "wires": [3]},
#         {"input_idx": [4], "func": "ry", "wires": [4]},
#         {"input_idx": [5], "func": "ry", "wires": [5]},
#         {"input_idx": [6], "func": "ry", "wires": [6]},
#         {"input_idx": [7], "func": "ry", "wires": [7]},
#         {"input_idx": [8], "func": "ry", "wires": [8]},
#         {"input_idx": [9], "func": "ry", "wires": [9]},
#     ],
#     "25_ry": [
#         {"input_idx": [0], "func": "ry", "wires": [0]},
#         {"input_idx": [1], "func": "ry", "wires": [1]},
#         {"input_idx": [2], "func": "ry", "wires": [2]},
#         {"input_idx": [3], "func": "ry", "wires": [3]},
#         {"input_idx": [4], "func": "ry", "wires": [4]},
#         {"input_idx": [5], "func": "ry", "wires": [5]},
#         {"input_idx": [6], "func": "ry", "wires": [6]},
#         {"input_idx": [7], "func": "ry", "wires": [7]},
#         {"input_idx": [8], "func": "ry", "wires": [8]},
#         {"input_idx": [9], "func": "ry", "wires": [9]},
#         {"input_idx": [10], "func": "ry", "wires": [10]},
#         {"input_idx": [11], "func": "ry", "wires": [11]},
#         {"input_idx": [12], "func": "ry", "wires": [12]},
#         {"input_idx": [13], "func": "ry", "wires": [13]},
#         {"input_idx": [14], "func": "ry", "wires": [14]},
#         {"input_idx": [15], "func": "ry", "wires": [15]},
#         {"input_idx": [16], "func": "ry", "wires": [16]},
#         {"input_idx": [17], "func": "ry", "wires": [17]},
#         {"input_idx": [18], "func": "ry", "wires": [18]},
#         {"input_idx": [19], "func": "ry", "wires": [19]},
#         {"input_idx": [20], "func": "ry", "wires": [20]},
#         {"input_idx": [21], "func": "ry", "wires": [21]},
#         {"input_idx": [22], "func": "ry", "wires": [22]},
#         {"input_idx": [23], "func": "ry", "wires": [23]},
#         {"input_idx": [24], "func": "ry", "wires": [24]},
#     ],
#     "25_ryrz": [
#         {"input_idx": [0], "func": "ry", "wires": [0]},
#         {"input_idx": [1], "func": "ry", "wires": [1]},
#         {"input_idx": [2], "func": "ry", "wires": [2]},
#         {"input_idx": [3], "func": "ry", "wires": [3]},
#         {"input_idx": [4], "func": "ry", "wires": [4]},
#         {"input_idx": [5], "func": "ry", "wires": [5]},
#         {"input_idx": [6], "func": "ry", "wires": [6]},
#         {"input_idx": [7], "func": "ry", "wires": [7]},
#         {"input_idx": [8], "func": "ry", "wires": [8]},
#         {"input_idx": [9], "func": "ry", "wires": [9]},
#         {"input_idx": [10], "func": "ry", "wires": [10]},
#         {"input_idx": [11], "func": "ry", "wires": [11]},
#         {"input_idx": [12], "func": "ry", "wires": [12]},
#         {"input_idx": [13], "func": "ry", "wires": [13]},
#         {"input_idx": [14], "func": "ry", "wires": [14]},
#         {"input_idx": [15], "func": "ry", "wires": [15]},
#         {"input_idx": [16], "func": "ry", "wires": [16]},
#         {"input_idx": [17], "func": "ry", "wires": [17]},
#         {"input_idx": [18], "func": "ry", "wires": [18]},
#         {"input_idx": [19], "func": "ry", "wires": [19]},
#         {"input_idx": [20], "func": "ry", "wires": [20]},
#         {"input_idx": [21], "func": "rz", "wires": [0]},
#         {"input_idx": [22], "func": "rz", "wires": [1]},
#         {"input_idx": [23], "func": "rz", "wires": [2]},
#         {"input_idx": [24], "func": "rz", "wires": [3]},
#     ],
#     "6x6_ryzxy": [
#         {"input_idx": [0], "func": "ry", "wires": [0]},
#         {"input_idx": [1], "func": "ry", "wires": [1]},
#         {"input_idx": [2], "func": "ry", "wires": [2]},
#         {"input_idx": [3], "func": "ry", "wires": [3]},
#         {"input_idx": [4], "func": "ry", "wires": [4]},
#         {"input_idx": [5], "func": "ry", "wires": [5]},
#         {"input_idx": [6], "func": "ry", "wires": [6]},
#         {"input_idx": [7], "func": "ry", "wires": [7]},
#         {"input_idx": [8], "func": "ry", "wires": [8]},
#         {"input_idx": [9], "func": "ry", "wires": [9]},
#         {"input_idx": [10], "func": "rz", "wires": [0]},
#         {"input_idx": [11], "func": "rz", "wires": [1]},
#         {"input_idx": [12], "func": "rz", "wires": [2]},
#         {"input_idx": [13], "func": "rz", "wires": [3]},
#         {"input_idx": [14], "func": "rz", "wires": [4]},
#         {"input_idx": [15], "func": "rz", "wires": [5]},
#         {"input_idx": [16], "func": "rz", "wires": [6]},
#         {"input_idx": [17], "func": "rz", "wires": [7]},
#         {"input_idx": [18], "func": "rz", "wires": [8]},
#         {"input_idx": [19], "func": "rz", "wires": [9]},
#         {"input_idx": [20], "func": "rx", "wires": [0]},
#         {"input_idx": [21], "func": "rx", "wires": [1]},
#         {"input_idx": [22], "func": "rx", "wires": [2]},
#         {"input_idx": [23], "func": "rx", "wires": [3]},
#         {"input_idx": [24], "func": "rx", "wires": [4]},
#         {"input_idx": [25], "func": "rx", "wires": [5]},
#         {"input_idx": [26], "func": "rx", "wires": [6]},
#         {"input_idx": [27], "func": "rx", "wires": [7]},
#         {"input_idx": [28], "func": "rx", "wires": [8]},
#         {"input_idx": [29], "func": "rx", "wires": [9]},
#         {"input_idx": [30], "func": "ry", "wires": [0]},
#         {"input_idx": [31], "func": "ry", "wires": [1]},
#         {"input_idx": [32], "func": "ry", "wires": [2]},
#         {"input_idx": [33], "func": "ry", "wires": [3]},
#         {"input_idx": [34], "func": "ry", "wires": [4]},
#         {"input_idx": [35], "func": "ry", "wires": [5]},
#     ],
#     "6x6_ryrz": [
#         {"input_idx": [0], "func": "ry", "wires": [0]},
#         {"input_idx": [1], "func": "ry", "wires": [1]},
#         {"input_idx": [2], "func": "ry", "wires": [2]},
#         {"input_idx": [3], "func": "ry", "wires": [3]},
#         {"input_idx": [4], "func": "ry", "wires": [4]},
#         {"input_idx": [5], "func": "ry", "wires": [5]},
#         {"input_idx": [6], "func": "ry", "wires": [6]},
#         {"input_idx": [7], "func": "ry", "wires": [7]},
#         {"input_idx": [8], "func": "ry", "wires": [8]},
#         {"input_idx": [9], "func": "ry", "wires": [9]},
#         {"input_idx": [10], "func": "ry", "wires": [10]},
#         {"input_idx": [11], "func": "ry", "wires": [11]},
#         {"input_idx": [12], "func": "ry", "wires": [12]},
#         {"input_idx": [13], "func": "ry", "wires": [13]},
#         {"input_idx": [14], "func": "ry", "wires": [14]},
#         {"input_idx": [15], "func": "ry", "wires": [15]},
#         {"input_idx": [16], "func": "ry", "wires": [16]},
#         {"input_idx": [17], "func": "ry", "wires": [17]},
#         {"input_idx": [18], "func": "ry", "wires": [18]},
#         {"input_idx": [19], "func": "ry", "wires": [19]},
#         {"input_idx": [20], "func": "ry", "wires": [20]},
#         {"input_idx": [21], "func": "rz", "wires": [0]},
#         {"input_idx": [22], "func": "rz", "wires": [1]},
#         {"input_idx": [23], "func": "rz", "wires": [2]},
#         {"input_idx": [24], "func": "rz", "wires": [3]},
#         {"input_idx": [25], "func": "rz", "wires": [4]},
#         {"input_idx": [26], "func": "rz", "wires": [5]},
#         {"input_idx": [27], "func": "rz", "wires": [6]},
#         {"input_idx": [28], "func": "rz", "wires": [7]},
#         {"input_idx": [29], "func": "rz", "wires": [8]},
#         {"input_idx": [30], "func": "rz", "wires": [9]},
#         {"input_idx": [31], "func": "rz", "wires": [10]},
#         {"input_idx": [32], "func": "rz", "wires": [11]},
#         {"input_idx": [33], "func": "rz", "wires": [12]},
#         {"input_idx": [34], "func": "rz", "wires": [13]},
#         {"input_idx": [35], "func": "rz", "wires": [14]},
#     ],
#     "6x6_ryrzrx": [
#         {"input_idx": [0], "func": "ry", "wires": [0]},
#         {"input_idx": [1], "func": "ry", "wires": [1]},
#         {"input_idx": [2], "func": "ry", "wires": [2]},
#         {"input_idx": [3], "func": "ry", "wires": [3]},
#         {"input_idx": [4], "func": "ry", "wires": [4]},
#         {"input_idx": [5], "func": "ry", "wires": [5]},
#         {"input_idx": [6], "func": "ry", "wires": [6]},
#         {"input_idx": [7], "func": "ry", "wires": [7]},
#         {"input_idx": [8], "func": "ry", "wires": [8]},
#         {"input_idx": [9], "func": "ry", "wires": [9]},
#         {"input_idx": [10], "func": "ry", "wires": [10]},
#         {"input_idx": [11], "func": "ry", "wires": [11]},
#         {"input_idx": [12], "func": "ry", "wires": [12]},
#         {"input_idx": [13], "func": "ry", "wires": [13]},
#         {"input_idx": [14], "func": "ry", "wires": [14]},
#         {"input_idx": [15], "func": "ry", "wires": [15]},
#         {"input_idx": [16], "func": "rz", "wires": [0]},
#         {"input_idx": [17], "func": "rz", "wires": [1]},
#         {"input_idx": [18], "func": "rz", "wires": [2]},
#         {"input_idx": [19], "func": "rz", "wires": [3]},
#         {"input_idx": [20], "func": "rz", "wires": [4]},
#         {"input_idx": [21], "func": "rz", "wires": [5]},
#         {"input_idx": [22], "func": "rz", "wires": [6]},
#         {"input_idx": [23], "func": "rz", "wires": [7]},
#         {"input_idx": [24], "func": "rz", "wires": [8]},
#         {"input_idx": [25], "func": "rz", "wires": [9]},
#         {"input_idx": [26], "func": "rz", "wires": [10]},
#         {"input_idx": [27], "func": "rz", "wires": [11]},
#         {"input_idx": [28], "func": "rz", "wires": [12]},
#         {"input_idx": [29], "func": "rz", "wires": [13]},
#         {"input_idx": [30], "func": "rz", "wires": [14]},
#         {"input_idx": [31], "func": "rz", "wires": [15]},
#         {"input_idx": [32], "func": "rx", "wires": [0]},
#         {"input_idx": [33], "func": "rx", "wires": [1]},
#         {"input_idx": [34], "func": "rx", "wires": [2]},
#         {"input_idx": [35], "func": "rx", "wires": [3]},
#     ],
#     "10x10_ryzxyzxyzxy": [
#         {"input_idx": [0], "func": "ry", "wires": [0]},
#         {"input_idx": [1], "func": "ry", "wires": [1]},
#         {"input_idx": [2], "func": "ry", "wires": [2]},
#         {"input_idx": [3], "func": "ry", "wires": [3]},
#         {"input_idx": [4], "func": "ry", "wires": [4]},
#         {"input_idx": [5], "func": "ry", "wires": [5]},
#         {"input_idx": [6], "func": "ry", "wires": [6]},
#         {"input_idx": [7], "func": "ry", "wires": [7]},
#         {"input_idx": [8], "func": "ry", "wires": [8]},
#         {"input_idx": [9], "func": "ry", "wires": [9]},
#         {"input_idx": [10], "func": "rz", "wires": [0]},
#         {"input_idx": [11], "func": "rz", "wires": [1]},
#         {"input_idx": [12], "func": "rz", "wires": [2]},
#         {"input_idx": [13], "func": "rz", "wires": [3]},
#         {"input_idx": [14], "func": "rz", "wires": [4]},
#         {"input_idx": [15], "func": "rz", "wires": [5]},
#         {"input_idx": [16], "func": "rz", "wires": [6]},
#         {"input_idx": [17], "func": "rz", "wires": [7]},
#         {"input_idx": [18], "func": "rz", "wires": [8]},
#         {"input_idx": [19], "func": "rz", "wires": [9]},
#         {"input_idx": [20], "func": "rx", "wires": [0]},
#         {"input_idx": [21], "func": "rx", "wires": [1]},
#         {"input_idx": [22], "func": "rx", "wires": [2]},
#         {"input_idx": [23], "func": "rx", "wires": [3]},
#         {"input_idx": [24], "func": "rx", "wires": [4]},
#         {"input_idx": [25], "func": "rx", "wires": [5]},
#         {"input_idx": [26], "func": "rx", "wires": [6]},
#         {"input_idx": [27], "func": "rx", "wires": [7]},
#         {"input_idx": [28], "func": "rx", "wires": [8]},
#         {"input_idx": [29], "func": "rx", "wires": [9]},
#         {"input_idx": [30], "func": "ry", "wires": [0]},
#         {"input_idx": [31], "func": "ry", "wires": [1]},
#         {"input_idx": [32], "func": "ry", "wires": [2]},
#         {"input_idx": [33], "func": "ry", "wires": [3]},
#         {"input_idx": [34], "func": "ry", "wires": [4]},
#         {"input_idx": [35], "func": "ry", "wires": [5]},
#         {"input_idx": [36], "func": "ry", "wires": [6]},
#         {"input_idx": [37], "func": "ry", "wires": [7]},
#         {"input_idx": [38], "func": "ry", "wires": [8]},
#         {"input_idx": [39], "func": "ry", "wires": [9]},
#         {"input_idx": [40], "func": "rz", "wires": [0]},
#         {"input_idx": [41], "func": "rz", "wires": [1]},
#         {"input_idx": [42], "func": "rz", "wires": [2]},
#         {"input_idx": [43], "func": "rz", "wires": [3]},
#         {"input_idx": [44], "func": "rz", "wires": [4]},
#         {"input_idx": [45], "func": "rz", "wires": [5]},
#         {"input_idx": [46], "func": "rz", "wires": [6]},
#         {"input_idx": [47], "func": "rz", "wires": [7]},
#         {"input_idx": [48], "func": "rz", "wires": [8]},
#         {"input_idx": [49], "func": "rz", "wires": [9]},
#         {"input_idx": [50], "func": "rx", "wires": [0]},
#         {"input_idx": [51], "func": "rx", "wires": [1]},
#         {"input_idx": [52], "func": "rx", "wires": [2]},
#         {"input_idx": [53], "func": "rx", "wires": [3]},
#         {"input_idx": [54], "func": "rx", "wires": [4]},
#         {"input_idx": [55], "func": "rx", "wires": [5]},
#         {"input_idx": [56], "func": "rx", "wires": [6]},
#         {"input_idx": [57], "func": "rx", "wires": [7]},
#         {"input_idx": [58], "func": "rx", "wires": [8]},
#         {"input_idx": [59], "func": "rx", "wires": [9]},
#         {"input_idx": [60], "func": "ry", "wires": [0]},
#         {"input_idx": [61], "func": "ry", "wires": [1]},
#         {"input_idx": [62], "func": "ry", "wires": [2]},
#         {"input_idx": [63], "func": "ry", "wires": [3]},
#         {"input_idx": [64], "func": "ry", "wires": [4]},
#         {"input_idx": [65], "func": "ry", "wires": [5]},
#         {"input_idx": [66], "func": "ry", "wires": [6]},
#         {"input_idx": [67], "func": "ry", "wires": [7]},
#         {"input_idx": [68], "func": "ry", "wires": [8]},
#         {"input_idx": [69], "func": "ry", "wires": [9]},
#         {"input_idx": [70], "func": "rz", "wires": [0]},
#         {"input_idx": [71], "func": "rz", "wires": [1]},
#         {"input_idx": [72], "func": "rz", "wires": [2]},
#         {"input_idx": [73], "func": "rz", "wires": [3]},
#         {"input_idx": [74], "func": "rz", "wires": [4]},
#         {"input_idx": [75], "func": "rz", "wires": [5]},
#         {"input_idx": [76], "func": "rz", "wires": [6]},
#         {"input_idx": [77], "func": "rz", "wires": [7]},
#         {"input_idx": [78], "func": "rz", "wires": [8]},
#         {"input_idx": [79], "func": "rz", "wires": [9]},
#         {"input_idx": [80], "func": "rx", "wires": [0]},
#         {"input_idx": [81], "func": "rx", "wires": [1]},
#         {"input_idx": [82], "func": "rx", "wires": [2]},
#         {"input_idx": [83], "func": "rx", "wires": [3]},
#         {"input_idx": [84], "func": "rx", "wires": [4]},
#         {"input_idx": [85], "func": "rx", "wires": [5]},
#         {"input_idx": [86], "func": "rx", "wires": [6]},
#         {"input_idx": [87], "func": "rx", "wires": [7]},
#         {"input_idx": [88], "func": "rx", "wires": [8]},
#         {"input_idx": [89], "func": "rx", "wires": [9]},
#         {"input_idx": [90], "func": "ry", "wires": [0]},
#         {"input_idx": [91], "func": "ry", "wires": [1]},
#         {"input_idx": [92], "func": "ry", "wires": [2]},
#         {"input_idx": [93], "func": "ry", "wires": [3]},
#         {"input_idx": [94], "func": "ry", "wires": [4]},
#         {"input_idx": [95], "func": "ry", "wires": [5]},
#         {"input_idx": [96], "func": "ry", "wires": [6]},
#         {"input_idx": [97], "func": "ry", "wires": [7]},
#         {"input_idx": [98], "func": "ry", "wires": [8]},
#         {"input_idx": [99], "func": "ry", "wires": [9]},
#     ],
#     "8x8_ryzxyzxy": [
#         {"input_idx": [0], "func": "ry", "wires": [0]},
#         {"input_idx": [1], "func": "ry", "wires": [1]},
#         {"input_idx": [2], "func": "ry", "wires": [2]},
#         {"input_idx": [3], "func": "ry", "wires": [3]},
#         {"input_idx": [4], "func": "ry", "wires": [4]},
#         {"input_idx": [5], "func": "ry", "wires": [5]},
#         {"input_idx": [6], "func": "ry", "wires": [6]},
#         {"input_idx": [7], "func": "ry", "wires": [7]},
#         {"input_idx": [8], "func": "ry", "wires": [8]},
#         {"input_idx": [9], "func": "ry", "wires": [9]},
#         {"input_idx": [10], "func": "rz", "wires": [0]},
#         {"input_idx": [11], "func": "rz", "wires": [1]},
#         {"input_idx": [12], "func": "rz", "wires": [2]},
#         {"input_idx": [13], "func": "rz", "wires": [3]},
#         {"input_idx": [14], "func": "rz", "wires": [4]},
#         {"input_idx": [15], "func": "rz", "wires": [5]},
#         {"input_idx": [16], "func": "rz", "wires": [6]},
#         {"input_idx": [17], "func": "rz", "wires": [7]},
#         {"input_idx": [18], "func": "rz", "wires": [8]},
#         {"input_idx": [19], "func": "rz", "wires": [9]},
#         {"input_idx": [20], "func": "rx", "wires": [0]},
#         {"input_idx": [21], "func": "rx", "wires": [1]},
#         {"input_idx": [22], "func": "rx", "wires": [2]},
#         {"input_idx": [23], "func": "rx", "wires": [3]},
#         {"input_idx": [24], "func": "rx", "wires": [4]},
#         {"input_idx": [25], "func": "rx", "wires": [5]},
#         {"input_idx": [26], "func": "rx", "wires": [6]},
#         {"input_idx": [27], "func": "rx", "wires": [7]},
#         {"input_idx": [28], "func": "rx", "wires": [8]},
#         {"input_idx": [29], "func": "rx", "wires": [9]},
#         {"input_idx": [30], "func": "ry", "wires": [0]},
#         {"input_idx": [31], "func": "ry", "wires": [1]},
#         {"input_idx": [32], "func": "ry", "wires": [2]},
#         {"input_idx": [33], "func": "ry", "wires": [3]},
#         {"input_idx": [34], "func": "ry", "wires": [4]},
#         {"input_idx": [35], "func": "ry", "wires": [5]},
#         {"input_idx": [36], "func": "ry", "wires": [6]},
#         {"input_idx": [37], "func": "ry", "wires": [7]},
#         {"input_idx": [38], "func": "ry", "wires": [8]},
#         {"input_idx": [39], "func": "ry", "wires": [9]},
#         {"input_idx": [40], "func": "rz", "wires": [0]},
#         {"input_idx": [41], "func": "rz", "wires": [1]},
#         {"input_idx": [42], "func": "rz", "wires": [2]},
#         {"input_idx": [43], "func": "rz", "wires": [3]},
#         {"input_idx": [44], "func": "rz", "wires": [4]},
#         {"input_idx": [45], "func": "rz", "wires": [5]},
#         {"input_idx": [46], "func": "rz", "wires": [6]},
#         {"input_idx": [47], "func": "rz", "wires": [7]},
#         {"input_idx": [48], "func": "rz", "wires": [8]},
#         {"input_idx": [49], "func": "rz", "wires": [9]},
#         {"input_idx": [50], "func": "rx", "wires": [0]},
#         {"input_idx": [51], "func": "rx", "wires": [1]},
#         {"input_idx": [52], "func": "rx", "wires": [2]},
#         {"input_idx": [53], "func": "rx", "wires": [3]},
#         {"input_idx": [54], "func": "rx", "wires": [4]},
#         {"input_idx": [55], "func": "rx", "wires": [5]},
#         {"input_idx": [56], "func": "rx", "wires": [6]},
#         {"input_idx": [57], "func": "rx", "wires": [7]},
#         {"input_idx": [58], "func": "rx", "wires": [8]},
#         {"input_idx": [59], "func": "rx", "wires": [9]},
#         {"input_idx": [60], "func": "ry", "wires": [0]},
#         {"input_idx": [61], "func": "ry", "wires": [1]},
#         {"input_idx": [62], "func": "ry", "wires": [2]},
#         {"input_idx": [63], "func": "ry", "wires": [3]},
#     ],
#     "8x2_ryz": [
#         {"input_idx": [0], "func": "ry", "wires": [0]},
#         {"input_idx": [1], "func": "ry", "wires": [1]},
#         {"input_idx": [2], "func": "ry", "wires": [2]},
#         {"input_idx": [3], "func": "ry", "wires": [3]},
#         {"input_idx": [4], "func": "ry", "wires": [4]},
#         {"input_idx": [5], "func": "ry", "wires": [5]},
#         {"input_idx": [6], "func": "ry", "wires": [6]},
#         {"input_idx": [7], "func": "ry", "wires": [7]},
#         {"input_idx": [8], "func": "rz", "wires": [0]},
#         {"input_idx": [9], "func": "rz", "wires": [1]},
#         {"input_idx": [10], "func": "rz", "wires": [2]},
#         {"input_idx": [11], "func": "rz", "wires": [3]},
#         {"input_idx": [12], "func": "rz", "wires": [4]},
#         {"input_idx": [13], "func": "rz", "wires": [5]},
#         {"input_idx": [14], "func": "rz", "wires": [6]},
#         {"input_idx": [15], "func": "rz", "wires": [7]},
#     ],
#     "16x1_ry": [
#         {"input_idx": [0], "func": "ry", "wires": [0]},
#         {"input_idx": [1], "func": "ry", "wires": [1]},
#         {"input_idx": [2], "func": "ry", "wires": [2]},
#         {"input_idx": [3], "func": "ry", "wires": [3]},
#         {"input_idx": [4], "func": "ry", "wires": [4]},
#         {"input_idx": [5], "func": "ry", "wires": [5]},
#         {"input_idx": [6], "func": "ry", "wires": [6]},
#         {"input_idx": [7], "func": "ry", "wires": [7]},
#         {"input_idx": [8], "func": "ry", "wires": [8]},
#         {"input_idx": [9], "func": "ry", "wires": [9]},
#         {"input_idx": [10], "func": "ry", "wires": [10]},
#         {"input_idx": [11], "func": "ry", "wires": [11]},
#         {"input_idx": [12], "func": "ry", "wires": [12]},
#         {"input_idx": [13], "func": "ry", "wires": [13]},
#         {"input_idx": [14], "func": "ry", "wires": [14]},
#         {"input_idx": [15], "func": "ry", "wires": [15]},
#     ],
# }

# class QFCModel(tq.QuantumModule):
#     class QLayer(tq.QuantumModule):
#         def __init__(self):
#             super().__init__()
#             self.n_wires = 4
#             self.random_layer = tq.RandomLayer(
#                 n_ops=50, wires=list(range(self.n_wires))
#             )

#             # gates with trainable parameters
#             self.rx0 = tq.RX(has_params=True, trainable=True)
#             self.ry0 = tq.RY(has_params=True, trainable=True)
#             self.rz0 = tq.RZ(has_params=True, trainable=True)
#             self.crx0 = tq.CRX(has_params=True, trainable=True)

#         def forward(self, qdev: tq.QuantumDevice):
#             self.random_layer(qdev)

#             # some trainable gates (instantiated ahead of time)
#             self.rx0(qdev, wires=0)
#             self.ry0(qdev, wires=1)
#             self.rz0(qdev, wires=3)
#             self.crx0(qdev, wires=[0, 2])

#             # add some more non-parameterized gates (add on-the-fly)
#             qdev.h(wires=3)  # type: ignore
#             qdev.sx(wires=2)  # type: ignore
#             qdev.cnot(wires=[3, 0])  # type: ignore
#             qdev.rx(
#                 wires=1,
#                 params=torch.tensor([0.1]),
#                 static=self.static_mode,
#                 parent_graph=self.graph,
#             )  # type: ignore

#     def __init__(self):
#         super().__init__()
#         self.n_wires = 4
#         self.encoder = tq.GeneralEncoder(encoder_op_list_name_dict["4x4_staggered_axes"])

#         # enc_key = "4x4_staggered_axes"
#         # print(f"Using encoder: {enc_key}")
#         # if enc_key not in encoder_op_list_name_dict:
#         #     raise KeyError(f"{enc_key} not found in encoder_op_list_name_dict; available keys: {list(encoder_op_list_name_dict.keys())}")
#         # self.encoder = tq.GeneralEncoder(encoder_op_list_name_dict[enc_key])
#         # exit()
#         self.q_layer = self.QLayer()
        

#         self.measure = tq.MeasureAll(tq.PauliZ)

#     def forward(self, x, use_qiskit=False):
#         qdev = tq.QuantumDevice(
#             n_wires=self.n_wires, bsz=x.shape[0], device=x.device, record_op=True
#         )

#         bsz = x.shape[0]
#         x = F.avg_pool2d(x, 6).view(bsz, 16)
#         devi = x.device

#         if use_qiskit:
#             # use qiskit to process the circuit
#             # create the qiskit circuit for encoder
#             self.encoder(qdev, x)  
#             op_history_parameterized = qdev.op_history
#             qdev.reset_op_history()
#             encoder_circs = op_history2qiskit_expand_params(self.n_wires, op_history_parameterized, bsz=bsz)

#             # create the qiskit circuit for trainable quantum layers
#             self.q_layer(qdev)
#             op_history_fixed = qdev.op_history
#             qdev.reset_op_history()
#             q_layer_circ = op_history2qiskit(self.n_wires, op_history_fixed)

#             # create the qiskit circuit for measurement
#             measurement_circ = tq2qiskit_measurement(qdev, self.measure)

#             # assemble the encoder, trainable quantum layers, and measurement circuits
#             assembled_circs = qiskit_assemble_circs(
#                 encoder_circs, q_layer_circ, measurement_circ
#             )

#             # call the qiskit processor to process the circuit
#             x0 = self.qiskit_processor.process_ready_circs(qdev, assembled_circs).to(  # type: ignore
#                 devi
#             )
#             x = x0

#         else:
#             # use torchquantum to process the circuit
#             self.encoder(qdev, x)
#             qdev.reset_op_history()
#             self.q_layer(qdev)
#             x = self.measure(qdev)

#         x = x.reshape(bsz, 2, 2).sum(-1).squeeze()
#         x = F.log_softmax(x, dim=1)

#         return x


# def train(dataflow, model, device, optimizer):
#     total_loss = 0.0
#     total_samples = 0
#     total_correct = 0

#     for feed_dict in dataflow["train"]:
#         inputs = feed_dict["image"].to(device)
#         targets = feed_dict["digit"].to(device)

#         outputs = model(inputs)
#         loss = F.nll_loss(outputs, targets)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print(f"loss: {loss.item()}", end="\r")

#         # accumulate metrics
#         bsz = targets.size(0)
#         total_loss += loss.item() * bsz
#         total_samples += bsz
#         preds = outputs.argmax(dim=1)
#         total_correct += preds.eq(targets).sum().item()

#     avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
#     accuracy = total_correct / total_samples if total_samples > 0 else 0.0
#     return avg_loss, accuracy


# def valid_test(dataflow, split, model, device, qiskit=False):
#     target_all = []
#     output_all = []
#     with torch.no_grad():
#         for feed_dict in dataflow[split]:
#             inputs = feed_dict["image"].to(device)
#             targets = feed_dict["digit"].to(device)

#             outputs = model(inputs, use_qiskit=qiskit)

#             target_all.append(targets)
#             output_all.append(outputs)
#         target_all = torch.cat(target_all, dim=0)
#         output_all = torch.cat(output_all, dim=0)

#     _, indices = output_all.topk(1, dim=1)
#     masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
#     size = target_all.shape[0]
#     corrects = masks.sum().item()
#     accuracy = corrects / size
#     loss = F.nll_loss(output_all, target_all).item()

#     # print(f"{split} set accuracy: {accuracy}")
#     # print(f"{split} set loss: {loss}")
#     return accuracy


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--static", action="store_true", help="compute with " "static mode"
#     )
#     parser.add_argument("--pdb", action="store_true", help="debug with pdb")
#     parser.add_argument("--qiskit-simulation", action="store_true", help="run on a real quantum computer")
#     parser.add_argument(
#         "--wires-per-block", type=int, default=2, help="wires per block int static mode"
#     )
#     parser.add_argument(
#         "--epochs", type=int, default=2, help="number of training epochs"
#     )

#     args = parser.parse_args()

#     if args.pdb:
#         import pdb

#         pdb.set_trace()

#     seed = 0
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     dataset = MNIST(
#         root="./mnist_data",
#         train_valid_split_ratio=[0.9, 0.1],
#         digits_of_interest=[0, 2],
#         n_test_samples=75,
#     )
#     dataflow = dict()

#     for split in dataset:
#         sampler = torch.utils.data.RandomSampler(dataset[split])
#         dataflow[split] = torch.utils.data.DataLoader(
#             dataset[split],
#             batch_size=100,
#             sampler=sampler,
#             pin_memory=True,
#             num_workers=4,
#         )


#     use_cuda = torch.cuda.is_available()
    


#     # device = "cpu"
#     device = torch.device("cuda" if use_cuda else "cpu")

#     model = QFCModel().to(device)
    
#     # exit()
#     n_epochs = args.epochs
#     optimizer = optim.Adam(model.parameters(), lr=5e-2, weight_decay=1e-4)
#     scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

#     for epoch in range(1, n_epochs + 1):
#         # train
#         print(f"Epoch {epoch}:")
#         # train(dataflow, model, device, optimizer)
        
#         train_loss, train_acc = train(dataflow, model, device, optimizer)
       
#         # print(f"learning rate: {optimizer.param_groups[0]['lr']}")

#         # valid
#         valid_acc = valid_test(dataflow, "valid", model, device)
#         print(f"lr: {optimizer.param_groups[0]['lr']:.4f}, Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Valid acc: {valid_acc:.4f}")
#         scheduler.step()

#     # test
#     test_acc = valid_test(dataflow, "test", model, device, qiskit=False)
#     print(f"Test acc: {test_acc:.4f}")

# if __name__ == "__main__":
#     main()




"""
MIT License

Copyright (c) 2020-present TorchQuantum Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import random
import numpy as np

import torchquantum as tq
from torchquantum.plugin import (
    tq2qiskit_measurement,
    qiskit_assemble_circs,
    op_history2qiskit,
    op_history2qiskit_expand_params,
)

from torchquantum.dataset import MNIST
from torch.optim.lr_scheduler import CosineAnnealingLR


class QFCModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(self.n_wires))
            )

            # gates with trainable parameters
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)

            # some trainable gates (instantiated ahead of time)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])

            # add some more non-parameterized gates (add on-the-fly)
            qdev.h(wires=3)  # type: ignore
            qdev.sx(wires=2)  # type: ignore
            qdev.cnot(wires=[3, 0])  # type: ignore
            qdev.rx(
                wires=1,
                params=torch.tensor([0.1]),
                static=self.static_mode,
                parent_graph=self.graph,
            )  # type: ignore

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_u3_h_rx"])

        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=x.shape[0], device=x.device, record_op=True
        )

        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)
        devi = x.device

        if use_qiskit:
            # use qiskit to process the circuit
            # create the qiskit circuit for encoder
            self.encoder(qdev, x)  
            op_history_parameterized = qdev.op_history
            qdev.reset_op_history()
            encoder_circs = op_history2qiskit_expand_params(self.n_wires, op_history_parameterized, bsz=bsz)

            # create the qiskit circuit for trainable quantum layers
            self.q_layer(qdev)
            op_history_fixed = qdev.op_history
            qdev.reset_op_history()
            q_layer_circ = op_history2qiskit(self.n_wires, op_history_fixed)

            # create the qiskit circuit for measurement
            measurement_circ = tq2qiskit_measurement(qdev, self.measure)

            # assemble the encoder, trainable quantum layers, and measurement circuits
            assembled_circs = qiskit_assemble_circs(
                encoder_circs, q_layer_circ, measurement_circ
            )

            # call the qiskit processor to process the circuit
            x0 = self.qiskit_processor.process_ready_circs(qdev, assembled_circs).to(  # type: ignore
                devi
            )
            x = x0

        else:
            # use torchquantum to process the circuit
            self.encoder(qdev, x)
            qdev.reset_op_history()
            self.q_layer(qdev)
            x = self.measure(qdev)

        x = x.reshape(bsz, 2, 2).sum(-1).squeeze()
        x = F.log_softmax(x, dim=1)

        return x


def train(dataflow, model, device, optimizer):
    for feed_dict in dataflow["train"]:
        inputs = feed_dict["image"].to(device)
        targets = feed_dict["digit"].to(device)

        outputs = model(inputs)
        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item()}", end="\r")


def valid_test(dataflow, split, model, device, qiskit=False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict["image"].to(device)
            targets = feed_dict["digit"].to(device)

            outputs = model(inputs, use_qiskit=qiskit)

            target_all.append(targets)
            output_all.append(outputs)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()

    print(f"{split} set accuracy: {accuracy}")
    print(f"{split} set loss: {loss}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--static", action="store_true", help="compute with " "static mode"
    )
    parser.add_argument("--pdb", action="store_true", help="debug with pdb")
    parser.add_argument("--qiskit-simulation", action="store_true", help="run on a real quantum computer")
    parser.add_argument(
        "--wires-per-block", type=int, default=2, help="wires per block int static mode"
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="number of training epochs"
    )

    args = parser.parse_args()

    if args.pdb:
        import pdb

        pdb.set_trace()

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = MNIST(
        root="./mnist_data",
        train_valid_split_ratio=[0.9, 0.1],
        digits_of_interest=[3, 6],
        n_test_samples=75,
    )
    dataflow = dict()

    for split in dataset:
        sampler = torch.utils.data.RandomSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=256,
            sampler=sampler,
            # num_workers=8,
            # pin_memory=True,
        )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = QFCModel().to(device)

    n_epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    if True:
        # optionally to switch to the static mode, which can bring speedup
        # on training
        model.q_layer.static_on(wires_per_block=args.wires_per_block)

    for epoch in range(1, n_epochs + 1):
        # train
        print(f"Epoch {epoch}:")
        train(dataflow, model, device, optimizer)
        print(optimizer.param_groups[0]["lr"])

        # valid
        valid_test(dataflow, "valid", model, device)
        scheduler.step()

    # test
    valid_test(dataflow, "test", model, device, qiskit=False)

    if args.qiskit_simulation:
        # run on Qiskit simulator and real Quantum Computers
        try:
            from qiskit import IBMQ
            from torchquantum.plugin import QiskitProcessor

            # firstly perform simulate
            print(f"\nTest with Qiskit Simulator")
            processor_simulation = QiskitProcessor(use_real_qc=False)
            model.set_qiskit_processor(processor_simulation)
            valid_test(dataflow, "test", model, device, qiskit=True)

            # then try to run on REAL QC
            backend_name = "ibmq_lima"
            print(f"\nTest on Real Quantum Computer {backend_name}")
            # Please specify your own hub group and project if you have the
            # IBMQ premium plan to access more machines.
            processor_real_qc = QiskitProcessor(
                use_real_qc=True,
                backend_name=backend_name,
                hub="ibm-q",
                group="open",
                project="main",
            )
            model.set_qiskit_processor(processor_real_qc)
            valid_test(dataflow, "test", model, device, qiskit=True)
        except ImportError:
            print(
                "Please install qiskit, create an IBM Q Experience Account and "
                "save the account token according to the instruction at "
                "'https://github.com/Qiskit/qiskit-ibmq-provider', "
                "then try again."
            )


if __name__ == "__main__":
    main()