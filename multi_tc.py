import numpy as np
import torch
import pairwise_tc
import ascii
from collections import Counter
import time
import einsum_benchmark

def build_sizes(term_A, term_B, shape_A, shape_B):
    sizes = {}
    sizes = dict(zip(term_A, shape_A))
    sizes.update(dict(zip(term_B, shape_B)))
    return sizes


def invoke_contraction_torch(A, B, mini_f_string):
    A_t = torch.from_numpy(A)
    B_t = torch.from_numpy(B)
    tic = time.time()
    C = (torch.einsum(mini_f_string, A_t, B_t)).numpy()
    toc = time.time()
    return C, toc-tic


def invoke_contraction_numpy(A, B, mini_f_string):
    tic = time.time()
    C = np.einsum(mini_f_string, A, B) 
    toc = time.time()
    return C, toc-tic
    
def invoke_contraction_np_mm( A, B,mini_f_string):
    tic = time.time()
    mini_inputs, mini_output = mini_f_string.split("->")
    mini_inputs = mini_inputs.split(",")
    term_A = mini_inputs[0]
    term_B = mini_inputs[1]
    sizes = build_sizes(term_A, term_B, A.shape, B.shape)
   
    C= pairwise_tc.pairwise_tc_np_mm(term_A, term_B, mini_output, A, B, sizes)
    
    if type(C) != np.ndarray:
        C = np.array(C)
    toc = time.time()
    return C, toc-tic

def invoke_contraction_custom( A, B,mini_f_string):
    tic = time.time()
    mini_inputs, mini_output = mini_f_string.split("->")
    mini_inputs = mini_inputs.split(",")
    term_A = mini_inputs[0]
    term_B = mini_inputs[1]
    sizes = build_sizes(term_A, term_B, A.shape, B.shape)
   
    C= pairwise_tc.pairwise_tc_custom(term_A, term_B, mini_output, A, B, sizes)
    
    if type(C) != np.ndarray:
        C = np.array(C)
    toc = time.time()
    return C, toc-tic


def prepare_contraction(mini_f_string, A, B, backend="custom"):
    methods = {
        "custom": invoke_contraction_custom,
        "torch": invoke_contraction_torch,
        "numpy": invoke_contraction_numpy,
        "np_mm": invoke_contraction_np_mm,
    }
    method = methods[backend]
    try:
        C, time_fragment = method(A, B, mini_f_string)
    except Exception as e:
        print(f"Error in {backend}: {e}")
        C, time_fragment = None, None
    return C, time_fragment



def multi_tc(path, tensors, format_string, backend = "custom"):
    ssa_path = einsum_benchmark.meta.runtime.to_ssa_path(path)
    format_string = format_string.replace(" ", "")
    annotated_path = einsum_benchmark.meta.runtime.to_annotated_ssa_path(format_string, ssa_path, True)
    length = len(tensors)
    #input, output = format_string.split("->")
    tic = time.time()
    time_fragment = 0
    for t_tuple in annotated_path:
        first = t_tuple[0]
        second = t_tuple[1]
        mini_f_string = t_tuple[2]
        A = tensors[first]
        B = tensors[second]   
        C, time_fragment_tmp = prepare_contraction(mini_f_string, A, B, backend)
        if first >= length:
            A = np.empty((1,), dtype=np.float32) 
        if second >= length:
            B = np.empty((1,), dtype=np.float32) 
        time_fragment += time_fragment_tmp
        tensors.append(C)
        
    toc = time.time()
    tensors = tensors[:length]
    
    return C, toc-tic, time_fragment



