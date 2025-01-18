import opt_einsum as oe
import einsum_benchmark
import f_path_operation as fo
import numpy as np
import ascii

def exp():
    
    instance = einsum_benchmark.instances["mc_2020_arjun_057"]
    instance_s = einsum_benchmark.instances["mc_2022_079"]

    s_opt_size = instance.paths.opt_size

    for backend in ["torch","custom", "numpy"]:
        print("************************** "+backend+" *********************************")
        C, term_C, time = fo.work_path(s_opt_size.path, instance.tensors, instance.format_string, backend)
        print("sum[OUTPUT]:", C.sum(), instance.result_sum)
        print("time: ", time)

    #"mc_2023_arjun_117"
    #"mc_2021_027"
    #"mc_2020_082"
    #"lm_batch_likelihood_sentence_4_8d"
    #lm_batch_likelihood_sentence_3_12d
    #str_matrix_chain_multiplication_100
    
    



def test_ascii():
    term_A = "ԲӞ̩՟ԥ"
    term_B = "Ձӏ̸Ձԥ"
    term_O = "abc"
    f_list = [term_A, term_B, term_O]
    print(f_list)
    f_list_new, char_dict = ascii.convert_to_ascii(f_list)
    print(f_list_new)
    print(char_dict)

    f_list = ascii.convert_ascii_back(f_list_new, char_dict)
    print(f_list)

exp()