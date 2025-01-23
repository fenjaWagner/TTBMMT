import numpy as np
import math
import time
import wrapper



def map_to_bmm(term_1, term_2, Tensor_1, Tensor_2, contract_idcs, batch_idcs, keep_1_idcs, keep_2_idcs, sizes):#full_term_dict, sizes):
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
     
    O_Tensor = wrapper.call_cpp_bmm(Tensor_1, Tensor_2)
    term_O = batch_idcs + keep_1_idcs + keep_2_idcs

    size_o = tuple([sizes[i] for i in term_O])
    
    O_Tensor = np.ascontiguousarray(O_Tensor.reshape(size_o))
    return O_Tensor, term_O