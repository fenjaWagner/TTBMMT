#import opt_einsum as oe
import einsum_benchmark
#import f_path_operation as fop
import f_path_operation_copy as fo
import numpy as np
import ascii

def exp():
    i_list = ["lm_batch_likelihood_sentence_4_8d", "lm_batch_likelihood_sentence_3_12d", "wmc_2023_035", "mc_2021_027", "mc_2022_079"]

    for stri in i_list:
        print(f"*************************************** {stri} *******************************")

        instance = einsum_benchmark.instances[stri]
        s_opt_size = instance.paths.opt_size
        
        for backend in ["torch","custom", "numpy", "np_mm"]:
            C, time, time_fragment = fo.work_path(s_opt_size.path, instance.tensors, instance.format_string, backend)
            
            print(f"backend {backend} + time {time} + fragment_time {time_fragment} + difference {time-time_fragment}")

        #"mc_2023_arjun_117" -> 23 s, np 200
        #"mc_2021_027"0.5, 0.9, 10.7
        #"mc_2020_082" zu groß
        #mc_2022_079 0.3, 0.46, 0.21

        #"lm_batch_likelihood_sentence_4_8d", 0.8, 27.2, 11.7
        #lm_batch_likelihood_sentence_3_12d -> 0.2, 4.2, 1,5
        #mc_rw_blasted_case1_b14_even3 custom: killed
        # mc_2021_arjun_171 custom: killed
        # wmc_2021_130 torch: killed
        # wmc_2023_035 0.9, 1.1, 0.6

    
def blocks():
    
    instance = einsum_benchmark.instances["lm_batch_likelihood_sentence_3_12d"]

    s_opt_size = instance.paths.opt_size
    clock = 0
    for i in range(5):
        C, time = fo.work_path(s_opt_size.path, instance.tensors, instance.format_string, "custom")
        print(time)
        clock += time
    print("time", clock/5)

def exp_dtypes():
    format_string = "aaabbbcc, cddeef -> cad"
    A = np.random.rand(3,3,3,4,4,4,2,2)
    B = np.random.rand(2,3,3,4,4,5)
    C = np.einsum(format_string, A, B)
    for ty in [np.int16, np.int32, np.int64, np.float32, np.float64]:
        print(f"************************** {ty} *************************************" )
        C_c, time = fo.work_path([(0,1)], [A,B], format_string, "custom")
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