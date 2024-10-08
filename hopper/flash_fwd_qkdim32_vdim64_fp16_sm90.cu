// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_<cutlass::half_t, 32, 64>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_qkdim32_vdim64<cutlass::half_t>(params, stream);
}
