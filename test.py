import torch
from torch import nn
import math
from fa_with_bias import flash_attn_with_kvcache_and_bias

def batch_mask_attention(q, k, v, mask, valid_length, group_num):
    """
    q: [bsz, head, qlen(=1), dim]
    k: [bsz*group*max_length*dim]
    v: [bsz*group*max_length*dim]
    mask: [bsz*group*max_length]
    valid_length: int
    group_num: int
    
    return:
    attn_lse: [bsz, head, qlen(=1)]
    attn_out: [bsz, head, qlen(=1), dim]
    """
    bsz = q.size(0)
    head_num = q.size(1)
    dim = q.size(-1)
    group_size = head_num // group_num
 
    # bsz, group_num, group_size, dim
    q = q.view(bsz, group_num, group_size, dim)
    # bsz, group_num, valid_length, dim
    k = k[:bsz*group_num*valid_length*dim].view(bsz, group_num, valid_length, dim)
    # bsz, group_num, valid_length, dim
    v = v[:bsz*group_num*valid_length*dim].view(bsz, group_num, valid_length, dim)
    # bsz, group_num, valid_length
    mask = mask[:bsz*group_num*valid_length].view(bsz, group_num, valid_length)
 
 
    dist = torch.einsum('bghd,bgcd->bghc', q, k)   # (bsz, group_num, group_size, valid_length)
    attn_score = dist / math.sqrt(dim)
    # attn_score = attn_score.masked_fill(mask.unsqueeze(-2), torch.finfo(torch.float16).min)
    attn_score = attn_score + mask.unsqueeze(-2)
 
    attn_lse = torch.logsumexp(attn_score, dim=-1)              # (bsz, group_num, group_size)
    attn_score = nn.functional.softmax(attn_score, dim=-1)      # (bsz, group_num, group_size, valid_length)
    attn_out = torch.einsum('bghc,bgcd->bghd', attn_score, v)   # (bsz, group_num, group_size, dim)
   
    attn_lse = attn_lse.view(bsz, 1, head_num)
    attn_out = attn_out.view(bsz, 1, head_num, dim)
    return attn_lse, attn_out


def batch_flash_attention(q, k, v, mask, valid_length, group_num):
    """
    q: [bsz, head, qlen(=1), dim]
    k: [bsz*group*max_length*dim]
    v: [bsz*group*max_length*dim]
    mask: [bsz*group*max_length]
    valid_length: int
    group_num: int
 
    return:
    attn_lse: [bsz, head, qlen(=1)], fp32
    attn_out: [bsz, qlen(=1), head, dim], fp16
    """
    bsz = q.size(0)
    dim = q.size(-1)
 
    k = k[:bsz*group_num*valid_length*dim].view(bsz, group_num, valid_length, dim)
    v = v[:bsz*group_num*valid_length*dim].view(bsz, group_num, valid_length, dim)
    mask = mask[:bsz*group_num*valid_length].view(bsz, group_num, 1, valid_length).repeat(1,1,4,1)
 
    attn_out, attn_lse= flash_attn_with_kvcache_and_bias(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        mask,
        return_softmax_lse=True,
    )
    return attn_lse, attn_out


# def flash_decoding(
#     q: torch.Tensor,  # [bsz, num_q_heads, qlen(=1), dim]
#     k: torch.Tensor,  # [bsz, num_k_heads, max_length, dim]
#     v: torch.Tensor,  # [bsz, num_k_heads, max_length, dim]
#     mask: torch.Tensor,  # [bsz, num_k_heads, max_length]
#     seqlens: torch.Tensor,  # [bsz, num_k_heads]
# ):
#     attn_out, attn_lse = flash_attn_with_kvcache(
#         q.transpose(1, 2),
#         k.transpose(1, 2),
#         v.transpose(1, 2),
#         return_softmax_lse=True,
#     )
#     return attn_lse, attn_out.transpose(1, 2)

# def flash_decoding_varlen(
#     q: torch.Tensor,  # [bsz, num_q_heads, qlen(=1), dim]
#     k: torch.Tensor,  # [bsz, num_k_heads, max_length, dim]
#     v: torch.Tensor,  # [bsz, num_k_heads, max_length, dim]
#     mask: torch.Tensor,  # [bsz, num_k_heads, max_length]
#     seqlens: torch.Tensor,  # [bsz, num_k_heads]
# ):
#     bsz, num_q_heads, _, dim = q.shape
#     _, num_k_heads, max_length, _ = k.shape
#     group_size = num_q_heads // num_k_heads
#     attn_out, attn_lse = flash_attn_with_kvcache(
#         q.reshape((bsz * num_k_heads, group_size, 1, dim)).transpose(1, 2),
#         k.reshape((bsz * num_k_heads, 1, max_length, dim)).transpose(1, 2),
#         v.reshape((bsz * num_k_heads, 1, max_length, dim)).transpose(1, 2),
#         cache_seqlens=seqlens.flatten(),
#         return_softmax_lse=True,
#     )
#     return attn_lse.reshape((bsz, num_q_heads, 1)), attn_out.transpose(1, 2).reshape((bsz, num_q_heads, 1, dim))


bsz = 1
group = 8
head = group * 4
max_length = 8000
valid_length = 8000
dim = 128
 
device = "cuda"
dtype = torch.float16
 
q = torch.randn((bsz, head, 1, dim), dtype=dtype, device=device)
k = torch.randn((bsz*group*max_length*dim,), dtype=dtype, device=device)
v = torch.randn((bsz*group*max_length*dim,), dtype=dtype, device=device)
mask = torch.randint(0, 2, (bsz*group*max_length,), dtype=torch.bool, device=device)
mask = torch.where(mask, float('-inf'), torch.tensor(0.0)).to(dtype)

mask_pro =  mask[:bsz*group*valid_length].view(bsz, group, 1, valid_length).repeat(1,1,4,1)
 
# print(mask)

_ , out_ref= batch_mask_attention(q, k, v, mask, valid_length, group)
_, out = batch_flash_attention(q, k, v, mask, valid_length, group)
print(out_ref.shape, out.shape)
print(out[0,:,0])
print(out_ref[0,:,0])

iters = 1000
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)


start.record()
for i in range(iters):
    out, _ = batch_flash_attention(q, k, v, mask, valid_length, group)
end.record()
torch.cuda.synchronize()
print("flash: ", start.elapsed_time(end) / iters)

start.record()
for i in range(iters):
    out_ref, _ = batch_mask_attention(q, k, v, mask, valid_length, group)
end.record()
torch.cuda.synchronize()
print("ref: ", start.elapsed_time(end) / iters)