#import opt_einsum as oe
import einsum_benchmark
#import f_path_operation as fop
import f_path_operation_copy as fo
import numpy as np
import ascii
import cgreedy
import csv

def initialize_writing(backend):
    """Writes the given data to a CSV file."""
    with open(backend+".csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Flops", "Max_Size", "Time", "Difference"])

def append_results_to_csv(backend, data):
    """Appends the given data to a CSV file without removing previous entries."""
    with open(backend+".csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)


def exp():
    #i_list = ["lm_batch_likelihood_sentence_4_8d", "lm_batch_likelihood_sentence_3_12d", "wmc_2023_035", "mc_2021_027", "mc_2022_079"]
    i_list = ["lm_batch_likelihood_sentence_4_8d"]
    for stri in i_list:
        print(f"*************************************** {stri} *******************************")

        instance = einsum_benchmark.instances[stri]
        s_opt_size = instance.paths.opt_size
        
        for backend in ["torch", "custom", "numpy", "np_mm"]:
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


def generate_random_problem(number_of_tensors, max_order, edge_order, number_of_selfe_edges,selfe_edge_order, number_of_single_summation_indices, min, max ):
    format_string, shapes= einsum_benchmark.generators.random.connected_hypernetwork(number_of_tensors = number_of_tensors, 
                                                                                            regularity = 3.5,
                                                                                            max_tensor_order = max_order,
                                                                                            max_edge_order = edge_order,
                                                                                            diagonals_in_hyper_edges= True,
                                                                                            number_of_output_indices = 0,
                                                                                            max_output_index_order = 3,
                                                                                            diagonals_in_output_indices = False,
                                                                                            number_of_self_edges = number_of_selfe_edges,
                                                                                            max_self_edge_order = selfe_edge_order,
                                                                                            number_of_single_summation_indices= number_of_single_summation_indices,
                                                                                            global_dim = False,
                                                                                            min_axis_size = min,
                                                                                            max_axis_size = max,
                                                                                            seed= None,
                                                                                            return_size_dict= False)
    print("generated")
    tensors = []
    for shape in shapes:
        tensors.append(np.random.random_sample(shape))

    print("tensors")
    
    format_string = format_string.replace(" ", "")
    
    #path, size_log2, flops_log10 = cgreedy.compute_path(format_string, *tensors, seed=1, minimize="size", max_repeats=1024,
    #                                        max_time=1.0, progbar=True, threshold_optimal=12, threads=0, is_linear=True)
    print("greedy")
    
    
    return format_string, tensors#, path, size_log2, flops_log10
    
    

def generate_random_experiments():

    for backend in ["torch", "custom", "numpy", "np_mm"]:
        initialize_writing(backend)

    succesful = 0
    unsuccesful = 0
    big_ones = []

    for max_order in range(6):
        print(max_order)
        for i in range(1):
            try: 
                print("startet")
                format_string, tensors= generate_random_problem(number_of_tensors = 2, max_order = max_order, edge_order = 3, number_of_selfe_edges = 1, selfe_edge_order = 4, number_of_single_summation_indices = 1, min = 2, max = 10)
                print("format_string", format_string),
                print(tensors[0].shape, tensors[1].shape)
                #, path, size_log2, flops_log10 = generate_random_problem(number_of_tensors = 2, max_order = max_order, edge_order = 3, number_of_selfe_edges = 1, selfe_edge_order = 4, number_of_single_summation_indices = 1, min = 2, max = 10)
                #print("flops: ", flops_log10)
                #print("size: ", size_log2)
                #if flops_log10 > 20:
                #    big_ones.append((format_string, tensors, path, size_log2, flops_log10))
                for backend in ["torch", "custom", "numpy", "np_mm"]:
                    C, time, time_fragment = fo.work_path([(0,1)], tensors, format_string, backend)
                        
                    #print(f"backend {backend} + time {time} + fragment_time {time_fragment} + difference {time-time_fragment}")
                    #with open(backend+".csv", mode='a', newline='') as file:
                    #    writer = csv.writer(file)
                    #    writer.writerow([flops_log10, size_log2, time, time-time_fragment])
                    #        
                succesful += 1
            except:
                print("too big")
        print("successful: ", succesful)


generate_random_experiments()