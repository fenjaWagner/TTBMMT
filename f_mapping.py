import torch 
import numpy as np
import mm 
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
    for i in term_set:
        if i not in output_set:
            if i not in other_term_set:
                single_set.add(i)
            else:
                contract_set.add(i)
    return single_set, contract_set




def create_index_dict(term_A, term_B, term_O) -> dict:
    index_dict = {}
    index_dict["output"] = useful_funcs.create_set(term_O)
    
    create_double_dict(term_A, "A", index_dict)
    create_double_dict(term_B, "B", index_dict)

    index_dict["single_A"], index_dict["contract"]= create_single_dict(index_dict["term_A"], index_dict["term_B"], index_dict["output"])
    index_dict["single_B"], dummy = create_single_dict(index_dict["term_B"], index_dict["term_A"], index_dict["output"])

    return index_dict
    
    

    