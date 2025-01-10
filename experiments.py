import opt_einsum as oe
import einsum_benchmark
import f_path_operation as fo
import numpy as np

instance = einsum_benchmark.instances["mc_2023_arjun_117"]

opt_size_path_meta = instance.paths.opt_size
print("Size optimized path")
print("log10[FLOPS]:", round(opt_size_path_meta.flops, 2))
print("log2[SIZE]:", round(opt_size_path_meta.size, 2))

C, term_C, time = fo.work_path(opt_size_path_meta.path, instance.tensors, instance.format_string, "custom")
#result = oe.contract(
#    instance.format_string, *instance.tensors, optimize=opt_size_path_meta.path
#)
print("sum[OUTPUT]:", C.sum(), instance.result_sum)

#str_matrix_chain_multiplication_100 (17)