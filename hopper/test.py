#python test.py  --test_bwd --causal --check_pytorch --debug --batch=4 --dim_qk=128 --dim_v=256 --bf16 --iters=1
import torch
import time
import torch.nn.functional as F
from torch import nn, einsum
import os
import argparse
import torch.nn as nn
import ctypes
from termcolor import colored
from einops import einsum, rearrange
# try:
#     from xformers.ops import memory_efficient_attention, LowerTriangularMask, MemoryEfficientAttentionCutlassOp
# except ModuleNotFoundError:
#     print("No Xformers Detected")

import math

# from customized_flash_attn import flash_attn_func
# from flash_attn import flash_attn_func as flash_attn2_func
from flash_attn import flash_attn_func 
# from flash_attn_interface import flash_attn_f3b2_func as flash_attn_func_hopper
from flash_attn_interface import flash_attn_func as flash_attn_func_hopper
# torch.manual_seed(55)



parser = argparse.ArgumentParser()
parser.add_argument('--test_bwd', action="store_true", default=False, help='')
parser.add_argument('--causal', action="store_true", default=False, help='')
parser.add_argument('--check_pytorch', action="store_true", default=False, help='')
parser.add_argument('--debug', action="store_true", default=False, help='')
parser.add_argument('--batch', type=int, default=2, help='')
parser.add_argument('--seqlen_q', type=int, default=2048, help='')
parser.add_argument('--seqlen_kv', type=int, default=2048, help='')
parser.add_argument('--nheads', type=int, default=20, help='')
parser.add_argument('--nheads_k', type=int, default=20, help='')
parser.add_argument('--dim_qk', type=int, default=128, help='')
parser.add_argument('--dim_v', type=int, default=128, help='')
parser.add_argument('--iters', type=int, default=1, help='')
parser.add_argument('--bf16', action="store_true", default=False, help='')

args = parser.parse_args()

batch = args.batch
seqlen_q = args.seqlen_q
seqlen_kv = args.seqlen_kv
nheads = args.nheads
nheads_k = args.nheads_k
dim_qk = args.dim_qk
dim_v = args.dim_v
iters = args.iters

dtype = torch.float16
if args.bf16:
    dtype = torch.bfloat16
    # assert args.operation == "smv2" or not args.flash_kernel

query = torch.randn([batch, nheads, seqlen_q, dim_qk], dtype=dtype, device='cuda:0')
key = 3 * torch.randn([batch, nheads_k, seqlen_kv, dim_qk], dtype=dtype, device='cuda:0')
value = 3 * torch.randn([batch, nheads_k, seqlen_kv, dim_v], dtype=dtype, device='cuda:0')

grad = torch.randn([batch, nheads, seqlen_q, dim_v], dtype=dtype, device='cuda:0')
bias = torch.randn([1], dtype=torch.float32, device='cuda:0')




query1 = query.detach().requires_grad_(True)
key1 = key.detach().requires_grad_(True)
value1 = value.detach().requires_grad_(True)

if args.test_bwd:
    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)


def debug(name,expect, actual, atol=1e-3, rtol=1e-3):
    all_close = torch.allclose(expect, actual, atol=atol, rtol=rtol)
    print(name + "  all_close={}".format(all_close))
    if not all_close:
        diff = (expect - actual).abs()
        print("all_close={}, max={}, min={}, mean={}".format(all_close, diff.max().item(), diff.min().item(), diff.mean().item()))
        max_indices  = torch.nonzero(diff == diff.max().item())
        first_index = tuple(max_indices[0].tolist())
        print(f"Index: {first_index}, expect: {expect[first_index]}, actual: {actual[first_index]}") 
        # print(actual[0,0])
        # print(expect[0,0])
        # if actual.shape[1] == 2:
        #     print(actual[0,1, :,:])
        #     print(expect[0,1, :,:])

    return all_close

print(query.shape, key.shape, value.shape)

class Attn(nn.Module):
    def __init__(self):
        super(Attn, self).__init__()

    def forward(self, q, k, v, causal=True):
        batch_size, nheads, seqlen_q, _ = q.shape
        nheads_k = k.shape[1]

        num_head_groups = nheads // nheads_k
        query = rearrange(q, "b (h g) n d -> b g h n d", g=num_head_groups)
        qk = einsum(query, k, "b g h n d, b h s d -> b g h n s")
        seqlen_kv = k.shape[2]
        # qk = q @ k.transpose(-1, -2)
        softmax_scale = 1.0 / math.sqrt(k.size(-1))
        torch.set_printoptions(threshold=10000)
        # print(qk[0,0, 32:, :5])
        if causal:
            causal_mask = torch.triu(torch.full((seqlen_q, seqlen_kv), float("-inf"), device=qk.device), 1)
            qk = qk + causal_mask.to(dtype=qk.dtype)
        # print(qk[0,0,:5, :10])
        attn = F.softmax(qk * softmax_scale, dim=-1)
        o = einsum(attn, v, "b g h n s, b h s d -> b g h n d")
        o = rearrange(o, "b g h n d -> b (h g) n d")
        # o = attn @ v
        return o

torch_model = Attn()


start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for i in range(iters):
    output = flash_attn_func_hopper(query1.transpose(1,2), key1.transpose(1,2), value1.transpose(1,2), causal=args.causal).transpose(1,2)
    # output = (flash_attn_func(query1.transpose(1,2), key1.transpose(1,2), value1.transpose(1,2), causal=args.causal)).transpose(1,2)
    
    # output = torch.nn.functional.scaled_dot_product_attention(query1, key1, value1, is_causal=args.causal)
    if args.test_bwd:
        output.backward(grad)
        dq = query1.grad
        dk = key1.grad
        dv = value1.grad
end.record()
end.synchronize()
latency = start.elapsed_time(end)/iters
print(colored("latency: "+str(latency),'green'))

if args.check_pytorch:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(iters):
        # torch_output = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=args.causal)

        # torch_output = torch_model(query, key, value, args.causal)

        # attn_bias = LowerTriangularMask()
        # query = query.flatten(0, 1)
        # key = key.flatten(0, 1)
        # value = value.flatten(0, 1)
        # torch_output = memory_efficient_attention(query, key, value, attn_bias, op=MemoryEfficientAttentionCutlassOp)
        # torch_output = torch_output.reshape(batch, nheads, seqlen_q, dim_v)

        torch_output = (flash_attn_func(query.transpose(1,2), key.transpose(1,2), value.transpose(1,2), causal=args.causal)).transpose(1,2)
        # torch_output = flash_attn_func_hopper(query.transpose(1,2), key.transpose(1,2), value.transpose(1,2), causal=args.causal).transpose(1,2)
    
        if args.test_bwd:
            torch_output.backward(grad, retain_graph=True)
            torch_dq = query.grad
            torch_dk = key.grad
            torch_dv = value.grad
    end.record()
    end.synchronize()
    torch_latency = start.elapsed_time(end)/iters
    print(colored("torch latency: "+str(torch_latency),'green'))
    if args.debug:
        debug("output",torch_output, output)
        if args.test_bwd:
            debug("dv",torch_dv, dv)
            debug("dk",torch_dk, dk)
            debug("dq",torch_dq, dq)

    print(colored("speedup:{}x".format(round(torch_latency / latency,1)), 'green'))








