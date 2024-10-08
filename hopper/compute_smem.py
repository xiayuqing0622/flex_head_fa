Br = 128
Bc = 96
d = 128
vd = 256
bwd_smem = Br * d *2 + Bc * d * 2+ Bc * vd  * 2 +  Br * vd + Br * Bc + Br * Bc + 8 * 8
fwd_smem = Br * d * 2 + Bc * d * 2 + Bc * vd * 2 + 8 * 8
print("bwd_smem: ", bwd_smem*2)
print("fwd_smem: ", fwd_smem*2)