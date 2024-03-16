import math
import torch
from torch.nn import functional as F
from torch.nn import ReLU
from torch.utils.cpp_extension import load

# Function to load the custom CUDA kernels as Python modules
def load_custom_kernels():
    global minimal_attn, exp_flash_attn, polymax_flash_attn, relu_flash_attn, sat_attn
    minimal_attn = load(name='minimal_attn', sources=['main.cpp', 'flash.cu'], extra_cuda_cflags=['-O2'])
    exp_flash_attn = load(name='exp_attn', sources=['main.cpp', 'flash_exp.cu'], extra_cuda_cflags=['-O2'])
    polymax_flash_attn = load(name='polymax_attn', sources=['main.cpp', 'flash_polymax.cu'], extra_cuda_cflags=['-O2'])
    relu_flash_attn = load(name='relu_attn', sources=['main.cpp', 'flash_relu.cu'], extra_cuda_cflags=['-O2'])
    sat_attn = load(name='sat_attn', sources=['main.cpp', 'flash_sat.cu'], extra_cuda_cflags=['-O2'])

# Function to reset and initialize inputs for each profiling session
def initialize_inputs():
    batch_size = 16
    n_head = 12
    seq_len = 64
    head_embd = 64
    q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
    k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
    v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
    return q, k, v

# Load custom CUDA kernels once
load_custom_kernels()

# Define manual attention for baseline comparison
def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

# Define software relu attention as a comparison point
def relu_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    relu = ReLU()
    att = relu(att)
    y = att @ v
    return y

# List of attention mechanisms for profiling
attention_mechanisms = {
    'manual attention': manual_attn,
    'software relu attention': relu_attn,
    'minimal flash attention': minimal_attn.forward,
    'flash exp sat attention': sat_attn.forward,
    'flash exp attention': exp_flash_attn.forward,
    'flash relu attention': relu_flash_attn.forward,
    'polymax attention': polymax_flash_attn.forward
}

# Profile each attention mechanism
for name, func in attention_mechanisms.items():
    print(f'=== Profiling {name} ===')
    q, k, v = initialize_inputs()  # Reset and initialize inputs
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        result = func(q, k, v)
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
    print('--------------------------------------------')

