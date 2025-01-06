import numpy as np
import f_sum_single_index
import f_single_trace
import f_map_to_bmm 
import try_path as t
from collections import Counter

from cgreedy import CGreedy, compute_path
import einsum_benchmark

def build_sizes(term_A, term_B, shape_A, shape_B):
    sizes = {}
    for char, size in zip(term_A, shape_A):
        if char in sizes: 
            if sizes[char] != size:
                raise Exception("Unconsistend sizes.")
        else:
            sizes[char] = size
            
        
    for char, size in zip(term_B, shape_B):
        if char in sizes: 
            if sizes[char] != size:
                
                raise Exception("Unconsistend sizes.")
        else:
            sizes[char] = size
    return sizes
    
def remove_double_indices(term, Tensor, histo, sizes):
    s_term = set()
    double = {}
    for char in term:
        if char in s_term:
            if char in double:
                double[char] += 1
            else:
                double[char] = 1
        else:
            s_term.add(char)

    #update histo, invoke single_sum
    if double:
        for char in double:
            histo[char] -= double[char]
        Tensor, term = f_single_trace.single_trace(term, Tensor, sizes)
       
    return Tensor, term

def find_single_indices(term_A, term_B, histo):
    set_A = set()
    single_A = set()
    single_B = set()
    contract = set()
    batch = set()

    # build set of chars of term_A, remove them from histogramm, find singles in A
    for char in term_A:
        histo[char] -= 1
        set_A.add(char)
        if histo[char] == 0:
            single_A.add(char)
    
    #find singles in B
    for char in term_B:
        container = histo[char]
        container -= 1
        if char in set_A:
            if container == 0:
                contract.add(char)
            else:
                batch.add(char)
        else:
            if container == 0:
                single_B.add(char)
        histo[char] = container

    return single_A, single_B, contract, batch

def invoke_contraction(term_A, term_B, A, B, histo):
    sizes = build_sizes(term_A, term_B, A.shape, B.shape)
    #remove double indices
    A, term_A = remove_double_indices(term_A, A, histo, sizes)
    B, term_B = remove_double_indices(term_B, B, histo, sizes)

    #remove single indices
    single_A, single_B, contract, batch = find_single_indices(term_A, term_B, histo)
    print(f"single A {single_A} \n single B {single_B} \n contract {contract}\n batch {batch}")
    if single_A:
        A, term_A = f_sum_single_index.remove_single_index(A, term_A, single_A, sizes)
    if single_B:
        B, term_B = f_sum_single_index.remove_single_index(B, term_B, single_B, sizes)

    #contract -> Tensor_new, term_new
    return f_map_to_bmm.map_to_bmm(term_A, term_B, A, B, contract, batch, sizes)

    

def work_path(ssa_path, tensors, format_string):
    format_string = format_string.replace(" ", "")
    histogramm = Counter(format_string)
    input, output = format_string.split("->")
    term_strings = input.split(",")
    for t_tuple in ssa_path:
        first = t_tuple[0]
        second = t_tuple[1]
        A = tensors[first]
        B = tensors[second]
        term_A = term_strings[first]
        term_B = term_strings[second]
        
        C, term_C = invoke_contraction(term_A, term_B, A, B, histogramm)
        tensors.append(C)
        term_strings.append(term_C)
        for char in term_C:
            histogramm[char] += 1

    return C, term_C


def test_second():
    format_string, shapes = t.rand_equation(5, 3, seed=12, d_min=2, d_max=10, n_out=2)
    format_string =format_string.replace("e", "d")
    
    format_string = 'cidhd,ddf,gia,bfc,hgm->ba'
    shapes = [(4, 6, 5, 3, 5), (5, 5, 2), (8, 6, 8), (3, 2, 4), (3, 8, 5)]
    tensors = [np.random.rand(*shape) for shape in shapes]
    
    path, size_log2, flops_log10 = compute_path(format_string, *tensors, seed=1, minimize="size", max_repeats=1024,
                                            max_time=1.0, progbar=True, threshold_optimal=12, threads=0, is_linear=True)

    ssa = einsum_benchmark.meta.runtime.to_ssa_path(path)
    C, term_C = work_path(ssa, tensors, format_string)
    test_t = np.einsum(format_string, tensors[0], tensors[1], tensors[2], tensors[3], tensors[4])
    print(term_C)
    print(C.shape)
    print(test_t.shape)
    print((C-np.transpose(test_t)).sum())


test_second()
