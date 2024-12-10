import f_sum_single_index
import f_single_trace
import useful_funcs
import numpy as np
import torch


def manage_single_tensor(tensor, term, tensor_name, full_term_dict, sizes):
    if full_term_dict["double_"+tensor_name]:
        tensor, term = f_single_trace.single_trace(term, tensor, sizes)
        print("new term double", term)
        print("shape double", tensor.shape)
        full_term_dict["term_"+tensor_name] = useful_funcs.create_set(term)
    if full_term_dict["single_"+tensor_name]:
        tensor,  term  = f_sum_single_index.remove_single_index(tensor, term, tensor_name, full_term_dict, sizes)
        print("new term single", term)
        full_term_dict["term_"+tensor_name] = useful_funcs.create_set(term)

    return tensor, term, full_term_dict



def test_manage():
    A = np.random.rand(5,5,4,3,5,3)
    A_term = "ijmlin"
    term_dict_full = {"double_A": {"i"},
                        "term_A": {"i", "j", "l", "m", "n"},
                      "single_A": {"i", "l", "m"}}

    sizes = {}
    for i in range(len(A_term)):
        sizes[A_term[i]] = A.shape[i]

    At = torch.from_numpy(A)
    C, new_term, dict = manage_single_tensor(A, A_term, "A",term_dict_full,  sizes)
    print("new_term: ", new_term)
    print("dict: ", dict)
    operand = "ijmlin->"+new_term
    print("operand", operand) 
    D = torch.einsum(operand, At)
    
    Ct = torch.from_numpy(C)
    
    print((Ct-D).sum())

test_manage()