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