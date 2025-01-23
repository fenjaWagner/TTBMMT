import numpy as np
import math
import time
import wrapper

def prepare_indices(term_A, term_B,  mini_output):
    set_A = set(term_A)
    set_B = set(term_B)
    output_set = set(mini_output)    
    
    batch = set_A & set_B & output_set
    contract = (set_A & set_B) - output_set
    keep_A = (set_A & output_set) - set_B
    keep_B = (set_B & output_set) - set_A

    if not (batch or contract or keep_A):
        keep_A = set_A
        #flag_A = True
    if not (batch or contract or keep_B):
        keep_B = set_B
        #flag_B = True
    
    
    return [''.join(set_l) for set_l in [batch, contract, keep_A, keep_B]]

def map_to_bmm(term_1, term_2, term_O, Tensor_1, Tensor_2, sizes):#full_term_dict, sizes):
    """Maps the contraction of two given tensors to the bmm.

    Args:
        term_1 (str): _description_
        term_2 (str): _description_
        Tensor_1 (np.array): _description_
        Tensor_2 (np.array): _description_
        sizes (dict): _description_

    Returns:
        _type_: _description_
    """

    """batch_idcs = ''.join(batch)
    contract_idcs = ''.join(contract)
    keep_1_idcs = ''.join(keep_1)
    keep_2_idcs = ''.join(keep_2)"""
    print(f"Term_1 {term_1}, Term_2 {term_2}, Term_O {term_O}")
    batch_idcs, contract_idcs, keep_1_idcs, keep_2_idcs = prepare_indices(term_1, term_2, term_O)
    print(f"o_1 {keep_1_idcs}")
    print(f"o_2 {contract_idcs}")
    print(f"Shape_1 {Tensor_1.shape}, Shape_2 {Tensor_2.shape}")
    print(f"sizes {sizes}")
    result_type = np.result_type(Tensor_1)
    if np.result_type(Tensor_2) != result_type:
        Tensor_2.astype(result_type)

    size_batch = math.prod([sizes[i] for i in batch_idcs])
    size_sum = math.prod([sizes[i] for i in contract_idcs])
    size_rest_1 = math.prod([sizes[i] for i in keep_1_idcs])
    size_rest_2 = math.prod([sizes[i] for i in keep_2_idcs])
    print(f"batch {size_batch}, contract {size_sum}, rest_1 {size_rest_1}, rest_2 {size_rest_2}")

    if term_1:
        Tensor_1 = np.einsum(term_1 +'->'+batch_idcs+keep_1_idcs+contract_idcs, Tensor_1)
    if term_2:
        Tensor_2 = np.einsum(term_2 +'->'+batch_idcs+contract_idcs+keep_2_idcs, Tensor_2)

    Tensor_1 = np.ascontiguousarray((Tensor_1).reshape((size_batch, size_rest_1, size_sum )))
    Tensor_2 = np.ascontiguousarray((Tensor_2).reshape((size_batch, size_sum, size_rest_2)))
     
    O_Tensor = wrapper.call_cpp_bmm(Tensor_1, Tensor_2)
    term_O_tmp = batch_idcs + keep_1_idcs + keep_2_idcs

    size_o = tuple([sizes[i] for i in term_O])
    
    O_Tensor = np.ascontiguousarray(O_Tensor.reshape(size_o))
    if term_O != term_O_tmp:
        O_Tensor = np.einsum(term_O_tmp+"->"+term_O, O_Tensor)

    return O_Tensor




def map_to_np_mm(term_1, term_2, term_O, Tensor_1, Tensor_2, sizes):#full_term_dict, sizes):
    """Maps the contraction of two given tensors to the bmm.

    Args:
        term_1 (str): _description_
        term_2 (str): _description_
        Tensor_1 (np.array): _description_
        Tensor_2 (np.array): _description_
        sizes (dict): _description_

    Returns:
        _type_: _description_
    """

    """batch_idcs = ''.join(batch)
    contract_idcs = ''.join(contract)
    keep_1_idcs = ''.join(keep_1)
    keep_2_idcs = ''.join(keep_2)"""
    batch_idcs, contract_idcs, keep_1_idcs, keep_2_idcs = prepare_indices(term_1, term_2, term_O)
    result_type = np.result_type(Tensor_1)
    if np.result_type(Tensor_2) != result_type:
        Tensor_2.astype(result_type)

    size_batch = math.prod([sizes[i] for i in batch_idcs])
    size_sum = math.prod([sizes[i] for i in contract_idcs])
    size_rest_1 = math.prod([sizes[i] for i in keep_1_idcs])
    size_rest_2 = math.prod([sizes[i] for i in keep_2_idcs])

    if term_1:
        Tensor_1 = np.einsum(term_1 +'->'+batch_idcs+keep_1_idcs+contract_idcs, Tensor_1)
    if term_2:
        Tensor_2 = np.einsum(term_2 +'->'+batch_idcs+contract_idcs+keep_2_idcs, Tensor_2)

    Tensor_1 = np.ascontiguousarray((Tensor_1).reshape((size_batch, size_rest_1, size_sum )))
    Tensor_2 = np.ascontiguousarray((Tensor_2).reshape((size_batch, size_sum, size_rest_2)))
     
    O_Tensor = Tensor_1 @ Tensor_2
    term_O_tmp = batch_idcs + keep_1_idcs + keep_2_idcs

    size_o = tuple([sizes[i] for i in term_O])
    
    O_Tensor = np.ascontiguousarray(O_Tensor.reshape(size_o))
    if term_O != term_O_tmp:
        O_Tensor = np.einsum(term_O_tmp+"->"+term_O, O_Tensor)

    return O_Tensor