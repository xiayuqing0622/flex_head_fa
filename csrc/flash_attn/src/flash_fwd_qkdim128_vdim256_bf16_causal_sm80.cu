// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_qkdim128_vdim256_sm80.h"

template<>
void run_mha_fwd_<cutlass::bfloat16_t, 128, 256, true>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_qkdim128_vdim256<cutlass::bfloat16_t, true>(params, stream);
}