import math

import torch
from torch.nn import functional as F
from torch.nn import ReLU
from torch.utils.cpp_extension import load

# Load the CUDA kernel as a python module
minimal_attn = load(name='minimal_attn', sources=['main.cpp', 'flash.cu'], extra_cuda_cflags=['-O2'])
exp_flash_attn = load(name='exp_attn', sources=['main.cpp', 'flash_exp.cu'], extra_cuda_cflags=['-O2'])
polymax_flash_attn = load(name='polymax_attn', sources=['main.cpp', 'flash_polymax.cu'], extra_cuda_cflags=['-O2'])
relu_flash_attn = load(name='relu_attn', sources=['main.cpp', 'flash_relu.cu'], extra_cuda_cflags=['-O2'])
sat_attn = load(name='sat_attn', sources=['main.cpp', 'flash_sat.cu'], extra_cuda_cflags=['-O2'])

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 16
n_head = 12
seq_len = 64
head_embd = 64

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

print('=== profiling manual attention ===')

# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    manual_result = manual_attn(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== profiling software relu attention ===')

# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.

def relu_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    relu = ReLU()
    att = relu(att)
    y = att @ v
    return y

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    relu_result = relu_attn(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
print('--------------------------------------------')

print('=== profiling minimal flash attention === ')

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    minimal_result = minimal_attn.forward(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== profiling flash exp sat attention === ')

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    sat_result = sat_attn.forward(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== profiling flash exp attention === ')

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    exp_flash_result = exp_flash_attn.forward(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== profiling flash relu attention === ')

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    relu_flash_result = relu_flash_attn.forward(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== profiling polymax attention === ')

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    polymax_flash_result = polymax_flash_attn.forward(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

# print('attn values sanity check:', torch.allclose(minimal_result, manual_result, relu_result, rtol=0, atol=1e-02))
