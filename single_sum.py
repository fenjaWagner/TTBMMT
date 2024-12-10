import torch 
import numpy as np
import useful_funcs

def create_sum_terms(term_input, term_output):
    sum_terms = []
    set_o = useful_funcs.create_set(term_output)

    for i in term_input:
        if i not in set_o:
            sum_terms.append(i)

    return sum_terms

def create_sum_shape(sum_dimensions, sizes):
    s_shape = []
    for i in sum_dimensions:
        s_shape.append(sizes[i])
    return s_shape
    
def single_sum(term_input: str , Tensor: np.array , sum_dimensions: str):
    term_dict = useful_funcs.create_index_dict(term_input)
    new_term = term_input.translate({ord(i): None for i in sum_dimensions})
    shape_complete = Tensor.shape
    shape_s_d = create_sum_shape(sum_dimensions, term_dict)
    shape_new = create_sum_shape(new_term, term_dict)

    summed_shape_complete = useful_funcs.sum_shape(shape_complete)
    summed_shape_new = useful_funcs.sum_shape(shape_new)

    Tensor_new = np.zeros(useful_funcs.calc_new_length(shape_new))
    

    # Idee: wie in trace mit iteratorn. Sinnvoll ohne transponieren?




    

