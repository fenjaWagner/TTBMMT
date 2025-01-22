import numpy as np
import torch
import f_map_to_bmm 
import ascii
import try_path as t
from collections import Counter
import time
import einsum_benchmark

def build_sizes(term_A, term_B, shape_A, shape_B):
    sizes = {}
    sizes = dict(zip(term_A, shape_A))
    sizes.update(dict(zip(term_B, shape_B)))
    return sizes

def prepare_indices(term_A, term_B,  histo):
    set_zero = set()
    output_set = histo["set_O"]
    for char in term_A + term_B:
        histo[char] -= 1
        if histo[char] == 0:
            set_zero.add(char)
        if histo[char] < 0:
            raise Exception("below zero, something went wrong here.")
    
    output_set = output_set - set_zero

    set_A = set(term_A)
    set_B = set(term_B)
    flag_A = False
    flag_B = False
    
    
    batch = set_A & set_B & output_set
    contract = (set_A & set_B) - output_set
    keep_A = (set_A & output_set) - set_B
    keep_B = (set_B & output_set) - set_A

    if not (batch or contract or keep_A):
        keep_A = set_A
        flag_A = True
    if not (batch or contract or keep_B):
        keep_B = set_B
        flag_B = True
    
    for char in batch.union(keep_A,keep_B):
        histo[char] += 1
        output_set.add(char)
    histo["set_O"] = output_set
   
    string_list = [''.join(set_l) for set_l in [batch, contract, keep_A, keep_B]]
    
    
    return [''.join(set_l) for set_l in [batch, contract, keep_A, keep_B]] + [flag_A, flag_B]
    


def invoke_contraction_torch(A, B, term_A, term_B, batch, contract, keep_A, keep_B, flag_A = False, flag_B = False):
    output = batch+keep_A+keep_B
    format_string = term_A +','+term_B+"->"+output
    A_t = torch.from_numpy(A)
    B_t = torch.from_numpy(B)
    C = (torch.einsum(format_string, A_t, B_t)).numpy()
    return C, output

def invoke_contraction_numpy(A, B, term_A, term_B, batch, contract, keep_A, keep_B, flag_A = False, flag_B = False):
    output = batch+keep_A+keep_B
    format_string = term_A +','+term_B+"->"+output
    C = np.einsum(format_string, A, B)
    return C, output

def invoke_contraction_custom( A, B, term_A, term_B, batch, contract, keep_A, keep_B, flag_A, flag_B):
    sizes = build_sizes(term_A, term_B, A.shape, B.shape)
    C, term_C = f_map_to_bmm.map_to_bmm(term_A, term_B, A, B, contract, batch, keep_A, keep_B, sizes)
    
    if type(C) != np.ndarray:
        C = np.array(C)
    return C, term_C



def prepare_contraction(term_A, term_B, A,B,histo, backend = "custom"):
    methods = {"custom": invoke_contraction_custom,
               "torch": invoke_contraction_torch,
               "numpy": invoke_contraction_numpy}
    method = methods[backend]
    [batch, contract, keep_A, keep_B, flag_A, flag_B] = prepare_indices(term_A, term_B, histo)
    [term_A, term_B, batch, contract, keep_A, keep_B ], char_dict = ascii.convert_to_ascii([ term_A, term_B,batch, contract, keep_A, keep_B])
    C, term_C = method(A, B,term_A, term_B, batch, contract, keep_A, keep_B, flag_A, flag_B)
    return C, ascii.convert_ascii_back([term_C], char_dict)[0]



    

def work_path(path, tensors, format_string, backend = "custom"):
    ssa_path = einsum_benchmark.meta.runtime.to_ssa_path(path)
    format_string = format_string.replace(" ", "")
    histogramm = Counter(format_string)
    output_set = set(histogramm)
    histogramm["set_O"] = output_set
    input, output = format_string.split("->")
    term_strings = input.split(",")
    tic = time.time()
    for t_tuple in ssa_path:
        first = t_tuple[0]
        second = t_tuple[1]
        A = tensors[first]
        B = tensors[second]
        
        term_A = term_strings[first]
        term_B = term_strings[second]
        
        C, term_C = prepare_contraction(term_A, term_B, A, B, histogramm, backend)
        tensors.append(C)
        term_strings.append(term_C)
        #print("new_term", term_C)

    toc = time.time()
    
    if term_C != output:
        C = np.einsum(term_C + "->" + output, C)

    
    return C, term_C, toc-tic


