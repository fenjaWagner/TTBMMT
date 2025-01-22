import opt_einsum as oe
import einsum_benchmark
import f_path_operation as fo
import numpy as np
import ascii

def exp():
    
    instance = einsum_benchmark.instances["mc_2023_arjun_117"]
    instance_s = einsum_benchmark.instances["mc_2022_079"]

    s_opt_size = instance.paths.opt_size

    for backend in ["custom"]:#["torch","custom", "numpy"]:
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
    
    

def exp_dtypes():
    format_string = "aaabbbcc, cddeef -> cad"
    A = np.random.rand(3,3,3,4,4,4,2,2)
    B = np.random.rand(2,3,3,4,4,5)
    C = np.einsum(format_string, A, B)
    for ty in [np.int16, np.int32, np.int64, np.float32, np.float64]:
        print(f"************************** {ty} *************************************" )
        C_c, term_C, time = fo.work_path([(0,1)], [A,B], format_string, "custom")
        print("time: ", time)
        print(np.allclose(C, C_c))



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