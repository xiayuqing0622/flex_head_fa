from flash_attention_triton_sparse import flash_attn_sparse_func
from flash_attention_triton import flash_attn_func
import torch
import time
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
b = 1
h = 1
l = 128
lk = 128

d = 64

q = torch.ones(b, l, h, d).half()
k = torch.ones(b, lk, h, d).half()
v = torch.ones(b, lk, h, d).half()
causal_mask = torch.randn(1, h, l, lk)
flash = True
# torch.save(torch.transpose(q, 1, 2),'q128.pt')
# torch.save(torch.transpose(k, 1, 2), 'k128.pt')
# torch.save(torch.transpose(v, 1, 2), 'v128.pt')
q = q.cuda()
k = k.cuda()
v = v.cuda()
def attention(q, k, v, causal_mask=None, rel=None):
    """
        q,k,v : b h g d
        rel: g g
        causal: (h) g g
    """
    sim = einsum('... i d, ... j d -> ... i j', q, k)
    # if rel is not None:
    #     sim = sim + rel
    if causal_mask is not None:
        sim = sim.masked_fill(causal_mask, float("-inf"))
    attn = F.softmax(sim, dim=-1, dtype=torch.float32).to(q.dtype)
    # attn = torch.nan_to_num(attn)
    out = einsum('... i j, ... j d -> ... i d', attn, v)
    return out
if flash:
    torch.cuda.synchronize()
    a = time.time()
    for i in range(100):
        start = time.time()
        out = flash_attn_func(q, k, v, None, False, None)
        # print(out)
        # out = flash_attn_func(q, k, v, causal_mask, False, None, 1)
        # out = flash_attn_sparse_func(q, k, v, causal_mask, True, None, 4)
        # torch.save(torch.transpose(out, 1, 2), 'out128.pt')
        # print(torch.transpose(out, 1, 2))
    torch.cuda.synchronize()
    print((time.time() - a)/100)
    # out = flash_attn_func(q, k, v, causal_mask, False, None)
else:
    q = torch.ones(b, h, l, d)
    k = torch.ones(b, h, lk, d)
    v = torch.ones(b, h, lk, d)
    mask_q = mask_k = torch.arange(l, device=q.device)
    causal_mask = mask_q[:, None] < mask_k[None, :]
    torch.cuda.synchronize()
    a = time.time()
    for i in range(100):
        attention(q,k,v,causal_mask)
    torch.cuda.synchronize()
    print((time.time() - a)/100)
    

