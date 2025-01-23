import numpy as np
import torch
import f_map_to_bmm_copy
import ascii
from collections import Counter
import time
import einsum_benchmark

def build_sizes(term_A, term_B, shape_A, shape_B):
    sizes = {}
    sizes = dict(zip(term_A, shape_A))
    sizes.update(dict(zip(term_B, shape_B)))
    return sizes


def invoke_contraction_torch(A, B, mini_f_string):#, flag_A = False, flag_B = False):
    A_t = torch.from_numpy(A)
    B_t = torch.from_numpy(B)
    C = (torch.einsum(mini_f_string, A_t, B_t)).numpy()
    return C

def invoke_contraction_numpy(A, B, mini_f_string):#, flag_A = False, flag_B = False):
    C = np.einsum(mini_f_string, A, B)
    return C

def invoke_contraction_np_mm( A, B,mini_f_string):#, flag_A, flag_B):
    mini_inputs, mini_output = mini_f_string.split("->")
    mini_inputs = mini_inputs.split(",")
    term_A = mini_inputs[0]
    term_B = mini_inputs[1]
    sizes = build_sizes(term_A, term_B, A.shape, B.shape)
    C= f_map_to_bmm_copy.map_to_bmm(term_A, term_B, mini_output, A, B, sizes)
    
    if type(C) != np.ndarray:
        C = np.array(C)
    return C


def invoke_contraction_custom( A, B,mini_f_string):#, flag_A, flag_B):
    mini_inputs, mini_output = mini_f_string.split("->")
    mini_inputs = mini_inputs.split(",")
    term_A = mini_inputs[0]
    term_B = mini_inputs[1]
    sizes = build_sizes(term_A, term_B, A.shape, B.shape)
    C= f_map_to_bmm_copy.map_to_bmm(term_A, term_B, mini_output, A, B, sizes)
    
    if type(C) != np.ndarray:
        C = np.array(C)
    return C



def prepare_contraction(mini_f_string, A,B, backend = "custom"):
    methods = {"custom": invoke_contraction_custom,
               "torch": invoke_contraction_torch,
               "numpy": invoke_contraction_numpy,
               "np_mm": invoke_contraction_np_mm}
    method = methods[backend]
    C = method(A, B,mini_f_string)#, flag_A, flag_B)
    return C


def work_path(path, tensors, format_string, backend = "custom"):
    ssa_path = einsum_benchmark.meta.runtime.to_ssa_path(path)
    format_string = format_string.replace(" ", "")
    annotated_path = einsum_benchmark.meta.runtime.to_annotated_ssa_path(format_string, ssa_path, True)
    input, output = format_string.split("->")
    tic = time.time()
    for t_tuple in annotated_path:
        first = t_tuple[0]
        second = t_tuple[1]
        mini_f_string = t_tuple[2]
        A = tensors[first]
        B = tensors[second]
        C= prepare_contraction(mini_f_string, A, B, backend)
        tensors.append(C)
        #term_strings.append(term_C)
        #print("new_term", term_C)

    toc = time.time()
    
    #if term_C != output:
    #    C = np.einsum(term_C + "->" + output, C)

    
    return C, toc-tic


