import torch 
import numpy as np
import re
import mm 

def find_double_indices(term):
    dim_set = set()
    single_indices = {}
    double_indices = {}

    for i in range(len(term)):
        dim = term[i]
        if i in dim_set:
            i_list = single_indices.pop(dim)
            double_indices[dim] = i_list.append(i)
        else:
            dim_set.add(dim)
            single_indices[dim] = [i]

    return single_indices, double_indices

def create_transpose_tuple(single_indices, double_indices):
    single_transpose = [single_indices[k] for k in single_indices]

    single_term = ""
    for k in single_indices:
        single_term += str(k)

    double_transpose = []
    for k in double_indices:
        double_transpose += double_indices[k]
    
    return single_transpose, double_transpose, single_term, double_indices



