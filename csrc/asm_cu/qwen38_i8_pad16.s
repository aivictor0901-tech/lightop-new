// SPDX-License-Identifier: Apache-2.0
// Bit-preserving A/scales padding to the launch-grid row count.
// The retained M24 gate/up route launches 32 rows before native INT8 GEMM.
.text
.amdgcn_target "amdgcn-amd-amdhsa--gfx936"
.protected qwen38_i8_pad16
.globl qwen38_i8_pad16
.p2align 8
.type qwen38_i8_pad16,@function
qwen38_i8_pad16:
  s_load_dwordx4 s[8:11], s[0:1], 0
  s_load_dwordx4 s[12:15], s[0:1], 16
  s_load_dwordx2 s[16:17], s[0:1], 32
  s_waitcnt lgkmcnt(0)
  s_mul_i32 s20, s2, s17
  s_lshl_b32 s21, s3, 12
  v_lshlrev_b32 v1, 4, v0
  v_add_u32 v1, s21, v1
  v_add_u32 v2, s20, v1
  v_cmp_lt_u32 vcc, v1, s17
  s_and_b64 exec, exec, vcc
  s_cbranch_execz .Lend_pad
  s_cmp_lt_u32 s2, s16
  s_cbranch_scc1 .Lload_pad
  v_mov_b32 v4, 0
  v_mov_b32 v5, 0
  v_mov_b32 v6, 0
  v_mov_b32 v7, 0
  s_branch .Lstore_pad
.Lload_pad:
  global_load_dwordx4 v[4:7], v2, s[8:9]
  s_waitcnt vmcnt(0)
.Lstore_pad:
  global_store_dwordx4 v2, v[4:7], s[12:13]
  s_cmp_eq_u32 s3, 0
  s_cbranch_scc0 .Lend_pad
  v_cmp_eq_u32 vcc, 0, v0
  s_and_b64 exec, exec, vcc
  s_cbranch_execz .Lend_pad
  s_lshl_b32 s22, s2, 2
  v_mov_b32 v2, s22
  s_cmp_lt_u32 s2, s16
  s_cbranch_scc1 .Lscale_load_pad
  v_mov_b32 v4, 0x3f800000
  s_branch .Lscale_store_pad
.Lscale_load_pad:
  global_load_dword v4, v2, s[10:11]
  s_waitcnt vmcnt(0)
.Lscale_store_pad:
  global_store_dword v2, v4, s[14:15]
.Lend_pad:
  s_endpgm
.Lsize_pad:
.size qwen38_i8_pad16, .Lsize_pad-qwen38_i8_pad16
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel qwen38_i8_pad16
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 40
  .amdhsa_user_sgpr_count 2
  .amdhsa_user_sgpr_private_segment_buffer 0
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 24
  .amdhsa_reserve_flat_scratch 0
  .amdhsa_reserve_xnack_mask 1
  .amdhsa_float_round_mode_32 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_dx10_clamp 1
  .amdhsa_ieee_mode 1
.end_amdhsa_kernel
.text
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.target: 'amdgcn-amd-amdhsa--gfx936'
amdhsa.kernels:
  - .name: qwen38_i8_pad16
    .symbol: qwen38_i8_pad16.kd
    .kernarg_segment_size: 40
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .sgpr_count: 28
    .vgpr_count: 8
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .offset: 8
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .offset: 16
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .offset: 24
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .offset: 32
        .size: 4
        .value_kind: by_value
      - .offset: 36
        .size: 4
        .value_kind: by_value
...
.end_amdgpu_metadata
