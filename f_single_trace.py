import numpy as np
import useful_funcs
import math
from collections import defaultdict
import time
import re
import mm 



def create_dict(term):
    term_dict = defaultdict(list)
    for i, char in enumerate(term):
        term_dict[char].append(i)
    return dict(term_dict)

"""def generate_new_term(term_dict, sizes):
    new_term = ""
    new_shape = []
    for k in term_dict:
        new_term += str(k)
        new_shape.append(sizes[k])
    return new_term, new_shape"""

def generate_new_term(term_dict, sizes):
    new_term = ''.join(str(k) for k in term_dict)
    new_shape = [sizes[k] for k in term_dict]
    return new_term, new_shape


def single_trace(term, A, sizes):#, single_idc, keep_idc):
    tic = time.time()

    #transpose_term = ''.join(list[keep_idc]+list[single_idc])
    #transpose_tuple = useful_funcs.transpose_tuple(term, transpose_term)
    #tensor = np.ascontiguousarray(np.transpose(tensor, transpose_tuple))

    term_dict = create_dict(term)
    new_term, new_shape = generate_new_term(term_dict, sizes)
    iterator = np.ndindex(tuple(new_shape))
    sum_old_shape = [math.prod(A.shape[i+1:]) for i in range(len(A.shape))]
    sum_new_shape = [math.prod(new_shape[i+1:]) for i in range(len(new_shape))]

    prod = math.prod(new_shape)
    A_new = np.zeros(prod)
    A = A.reshape(-1)
    
    for index in iterator:
        pos_A_old, pos_A_new = useful_funcs.calc_positions(index, sum_new_shape, sum_old_shape,  new_term,term_dict)
        A_new[pos_A_new] = A[pos_A_old]
            
    A_new = A_new.reshape(tuple(new_shape))
    toc = time.time()
    print(f"trace {toc-tic}")
    return A_new, new_term
