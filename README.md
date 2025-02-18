# FlexHeadFA

This repository is a fork of the [FlashAttention](https://github.com/Dao-AILab/flash-attention) main repo. It extends the official implementation to support FlashAttention with flexible head dimensions.

All configurations in FlashAttention-2 are supported. Besides, we have supported:

- FlashAttention-2 with QKHeadDim=32, VHeadDim=64 
- FlashAttention-2 with QKHeadDim=64, VHeadDim=128
- FlashAttention-2 with QKHeadDim=96, VHeadDim=192
- FlashAttention-2 with QKHeadDim=128, VHeadDim=256
- FlashAttention-2 with QKHeadDim=192, VHeadDim=128
- FLashAttention-2 with not equal num_heads_k and num_heads_v, such as (num_heads_q, num_heads_k, num_heads_v) = (32, 4, 16)

For headdim not supported, you can use the autotuner to generate the implementation. Details are in `autotuner.md`.

Feel free to tell us what else you need. We might support it soon. :)


## Installation

The requirements is the same as [FlashAttention-2](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features)

To install:
```sh
pip install flex-head-fa --no-build-isolation
```
Alternatively you can compile from source:

```sh
BUILD_LOCALLY=TRUE python setup.py install
```

The usage remains the same as FlashAttention-2. You only need to replace `flash_attn` with `flex_head_fa`, as shown below:


```python
from flex_head_fa import flash_attn_func, flash_attn_with_kvcache
```

We are also developing FlexHeadFA based on the lastest FLashAttention-3.
Currently, besides all configurations in FlashAttention-3, we also support

- FlashAttention-3 with QKHeadDim=32, VHeadDim=64
- FlashAttention-3 with QKHeadDim=64, VHeadDim=128
- FlashAttention-3 forward + FlashAttention-2 backward with QKHeadDim=128, VHeadDim=256 (FlashAttention-3 backward is under development)

Try it with:

```sh
cd hopper
python setup.py install
```

Usage:

```python
from flash_attn_interface import flash_attn_func # FlashAttention-3 forward+backward
from flash_attn_interface import flash_attn_f3b2_func as flash_attn_func # FlashAttention-3 forward + FlashAttention-2 backward 
```

## Performance of FlexHeadFA

We test the performance speedup compare to padding qk&v hidden_dim on A100.

We display FlexHeadFA speedup using these parameters:

- (qk dim, v_dim): (32,64), (64,128), (128,256); qk hidden dimension 2048 (i.e. 64, 32 or 16 heads).
- Sequence length 512, 1k, 2k, 4k, 8k, 16k.
- Batch size set to 16k / seqlen.

### Speedup

![Custom-flash-attn](assets/Customflash2_a100_fwd_bwd_benchmark.png)


## When you encounter issues

This new release of FlexHeadFA has been tested on several GPT-style
models, mostly on A100 GPUs.

If you encounter bugs, please open a GitHub Issue!


