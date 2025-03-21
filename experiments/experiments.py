#import opt_einsum as oe
import einsum_benchmark
#import f_path_operation as fop
import multi_tc as fo
import numpy as np
import ascii
import cgreedy
import csv
import json

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

def save_dictionary(filename, data):
    """Save a dictionary to a JSON file."""
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def exp():
    #i_list = ["lm_batch_likelihood_sentence_4_8d", "lm_batch_likelihood_sentence_3_12d", "wmc_2023_035", "mc_2021_027", "mc_2022_079"]
    i_list = ["mc_2020_arjun_046"]
    for stri in i_list:
        print(f"*************************************** {stri} *******************************")

        instance = einsum_benchmark.instances[stri]
        s_opt_size = instance.paths.opt_size
        dict = {}
        for backend in ["np_mm"]:#, "custom", "numpy", "np_mm"]:
            C, time, time_fragment , sizes= fo.multi_tc(s_opt_size.path, instance.tensors, instance.format_string, backend)
            
            print(f"backend {backend} + time {time} + fragment_time {time_fragment} + difference {time-time_fragment}, sum {C.sum()}, instance_s {instance.result_sum}")
            #print(sizes)
            size_prod = [max(sizes[i][0], sizes[i][1]) for i in range(0, len(sizes))]
            print(max(size_prod))
            max_flops = [m*m for m in size_prod]
            print(max(max_flops))
        #save_dictionary("dict_mc_2022_167.txt")


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
        C, time = fo.multi_tc(s_opt_size.path, instance.tensors, instance.format_string, "custom")
        print(time)
        clock += time
    print("time", clock/5)




def benchmark_experiments():
    
    #instance_list =  ["mc_2022_167", "mc_2022_079", "wmc_2021_130", "wmc_2023_035", "str_matrix_chain_multiplication_1000", "str_mps_varying_inner_product_2000", "wmc_2023_152", "str_mps_varying_inner_product_200", "mc_2023_002", "str_matrix_chain_multiplication_100", "lm_batch_likelihood_sentence_4_4d", "lm_batch_likelihood_brackets_4_4d", "mc_2023_188", "lm_batch_likelihood_sentence_3_12d", "mc_2020_017", "lm_batch_likelihood_sentence_4_8d", "mc_2023_arjun_117", "mc_2021_027",  "mc_rw_blasted_case1_b14_even3", "str_nw_peps_closed_333", "wmc_2023_141", "mc_2020_arjun_046", "mc_2020_arjun_057", "mc_2021_arjun_171", "str_nw_mera_closed_120", "lm_batch_likelihood_sentence_4_12d", "str_nw_mera_open_26", "rnd_mixed_08"]
    #instance_list = ["mc_2020_arjun_102", "mc_2021_065", "mc_rw_c7552.isc", "lm_first_last_sentence_4_14c", "lm_batch_likelihood_sentence_4_16d", "lm_batch_likelihood_sentence_4_16c", "gm_queen5_5_3.wcsp", "mc_2022_087", "mc_2022_025", "wmc_2023_036", "mc_2022_029", "mc_2021_075", "rnd_mixed_10", "rnd_oe_02", "str_ctg_lattice_02", "gm_pedigree38", "lm_batch_likelihood_sentence_4_18c", "gm_1kp6"]
    instance_list = ["mc_2022_167", "mc_2020_arjun_102", "wmc_2021_130", "wmc_2023_141", "wmc_2023_152", "lm_batch_likelihood_sentence_4_4d", "lm_batch_likelihood_sentence_4_12d"]
    with open("einsum_benchmark.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["instance", "flops", "max_size","custom" , "numpy", "np_mm", "torch"])
    for instance_name in instance_list:
        
        print(f"*************************************** doing {instance_name} *******************************")
        try: 
            instance = einsum_benchmark.instances[instance_name]
            s_opt_size = instance.paths.opt_size
            flops = s_opt_size.flops
            size = s_opt_size.size
            times = [instance_name, str(flops), str(size)]
            for backend in ["custom" , "numpy", "np_mm", "torch"]:
                print(backend)
                try: 
                    C, time, time_fragment = fo.multi_tc(s_opt_size.path, instance.tensors, instance.format_string, backend)
                    if backend == "torch":
                        times.append(time_fragment)
                        
                    else:
                        times.append(time)
                        print(time)
                except:
                    print(backend +" was killed.")
                    times.append("killed")
            
            with open("einsum_benchmark.csv", mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(times)
        
        except:
            print("killed on the way")
            

exp()




