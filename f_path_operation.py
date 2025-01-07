import numpy as np
import f_sum_single_index
import f_single_trace
import f_map_to_bmm 
import try_path as t
import useful_funcs
from collections import Counter
import time
import opt_einsum as oe

from cgreedy import CGreedy, compute_path
import einsum_benchmark

def build_sizes(term_A, term_B, shape_A, shape_B):
    sizes = {}
    sizes = dict(zip(term_A, shape_A))
    sizes.update(dict(zip(term_B, shape_B)))
    return sizes
    
def remove_double_indices(term, Tensor, histo, sizes):
    s_term = set()
    keep = set()
    double = {}
    
    for char in term:
        if term.count(char) == 1:
            keep.add(char)
        else: 
            double[char] = term.count(char)
    #update histo, invoke single_sum
    if double:
        for char in double:
            histo[char] -= double[char]
        tmp = term
        term = ''.join(keep)+''.join(double.keys())
        #Tensor, term = f_single_trace.single_trace(term, Tensor, sizes)
        Tensor = np.einsum(tmp +"->"+term, Tensor)
    return Tensor, term

def find_single_indices(term_A, term_B, histo):
    set_A = set()
    single_A = set()
    single_B = set()
    keep_A = set()
    keep_B = set()
    contract = set()
    batch = set()

    # build set of chars of term_A, remove them from histogramm, find singles in A
    for char in set(term_A+term_B):
        if histo[char]==1:
            histo[char] -= 1
            if char in term_A: single_A.add(char)      
            else: single_B.add(char)
        elif histo[char] == 2 and char in term_A and char in term_B:
            histo[char] -= 2
            contract.add(char)
        elif histo[char] > 2 and char in term_A and char in term_B:
            histo[char] -= 2
            batch.add(char)
        else:
            histo[char] -= 1
            if char in term_A:
                keep_A.add(char)
            else:
                keep_B.add(char)
            
        
    return single_A, single_B, keep_A, keep_B, contract, batch

def invoke_contraction(term_A, term_B, A, B, histo):
    sizes = build_sizes(term_A, term_B, A.shape, B.shape)
    single_A, single_B, keep_A, keep_B, contract, batch = find_single_indices(term_A, term_B, histo)
    
    # remove traces
    A, term_A = remove_double_indices(term_A, A, histo, sizes)
    B, term_B = remove_double_indices(term_B, B, histo, sizes)

    # sum over single indices
    #print(f"single A {single_A} \n single B {single_B} \n contract {contract}\n batch {batch}")
    if single_A:
        new_term = ''.join(keep_A)+''.join(batch)+''.join(contract)
        A = np.einsum(term_A + "->"+ new_term, A)
        term_A = new_term
        #A, term_A = f_sum_single_index.remove_single_index(A, term_A, single_A, sizes)
    if single_B:
        new_term = ''.join(keep_B)+''.join(batch)+''.join(contract)
        A = np.einsum(term_B + "->"+ new_term, B)
        term_B = new_term
        
        #B, term_B = f_sum_single_index.remove_single_index(B, term_B, single_B, sizes)

    #contract -> Tensor_new, term_new
    return f_map_to_bmm.map_to_bmm(term_A, term_B, A, B, contract, batch, keep_A, keep_B, sizes)

    

def work_path(ssa_path, tensors, format_string):
    format_string = format_string.replace(" ", "")
    histogramm = Counter(format_string)
    input, output = format_string.split("->")
    term_strings = input.split(",")
    for t_tuple in ssa_path:
        print("*"*20)
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
    if term_C != output:
        C = np.transpose(C,tuple(term_C.index(i) for i in output))
    return C, term_C


def test_second():
    format_string, shapes = t.rand_equation(20, 3, seed=7, d_min=2, d_max=7, n_out=4)      
    #format_string =format_string.replace("e", "d")
    
    format_string = 'cidhd,ddf,gia,bfc,hgm->ba'
    shapes = [(4, 6, 5, 3, 5), (5, 5, 2), (8, 6, 8), (3, 2, 4), (3, 8, 5)]
    tensors = [np.random.rand(*shape) for shape in shapes]
    tic = time.time()
    path, size_log2, flops_log10 = compute_path(format_string, *tensors, seed=1, minimize="size", max_repeats=1024,
                                            max_time=1.0, progbar=True, threshold_optimal=12, threads=0, is_linear=True)

    ssa = einsum_benchmark.meta.runtime.to_ssa_path(path)
    
    C, term_C = work_path(ssa, tensors, format_string)
    #C, term_C = work_path(path, tensors, format_string)
    
    toc = time.time()
    print(f"Time mine: {toc-tic}")

    #click = time.time()
    #test_t = np.einsum(format_string, tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], tensors[6], tensors[7], tensors[8], tensors[9])
    #clock = time.time()
    #print(f"Time numpy: {clock-click}")

    click = time.time()
    test_opt_eins = oe.contract(format_string, tensors[0], tensors[1], tensors[2], tensors[3], tensors[4])
    #test_opt_eins = oe.contract(format_string,tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], tensors[6], tensors[7], tensors[8], tensors[9],
    #                            tensors[10], tensors[11], tensors[12], tensors[13], tensors[14], tensors[15], tensors[16], tensors[17], tensors[18], tensors[19] )
    clock = time.time()
    print(f"Time oe: {clock-click}")
    
    print(term_C)
    print(C.shape)
    #print(test_t.shape)
    #print((C-np.transpose(test_t)).sum())
    #print(f"OE: {(C-test_opt_eins).sum()}")
    print(f"\n\n{np.allclose(C, test_opt_eins)}")


test_second()
