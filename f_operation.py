import torch 
import numpy as np
import f_single_engine
import f_map_to_bmm 
import useful_funcs

def create_double_dict(term, term_name, index_dict):
    term_set = set()
    double = set()
    for i in term:
        if i not in term_set:
            term_set.add(i)
        else: 
            double.add(i)
    index_dict["term_"+term_name] = term_set
    index_dict["double_"+term_name] = double


def create_single_dict(term_set, other_term_set, output_set):
    single_set = set()
    contract_set = set()
    batch_set = set()
    for i in term_set:
        if i not in output_set:
            if i not in other_term_set:
                single_set.add(i)
            else:
                contract_set.add(i)
        else:
            if i in other_term_set:
                batch_set.add(i)
    return single_set, contract_set, batch_set


def create_index_dict(term_A, term_B, term_O) -> dict:
    index_dict = {}
    index_dict["output"] = useful_funcs.create_set(term_O)
    
    create_double_dict(term_A, "A", index_dict)
    create_double_dict(term_B, "B", index_dict)

    index_dict["single_A"], index_dict["contract"], index_dict["batch"]= create_single_dict(index_dict["term_A"], index_dict["term_B"], index_dict["output"])
    index_dict["single_B"], dummy_c, dummy_b = create_single_dict(index_dict["term_B"], index_dict["term_A"], index_dict["output"])

    return index_dict


def double_engine(tensor_A, tensor_B, term_A, term_B, term_O):
    sizes = {}
    full_term_dict = create_index_dict(term_A, term_B, term_O)
    tensor_A, term_A, full_term_dict = f_single_engine.manage_single_tensor(tensor_A, term_A, "A", full_term_dict, sizes)
    tensor_B, term_B, full_term_dict = f_single_engine.manage_single_tensor(tensor_B, term_B, "B", full_term_dict, sizes)

    product_tensor, product_term = f_map_to_bmm.map_to_bmm(term_A, term_B, tensor_A, tensor_B, full_term_dict, sizes)
    
    return product_tensor, product_term


    
    

    