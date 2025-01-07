#import torch 
import numpy as np
import mm 
import useful_funcs
import math
import time


def map_to_bmm(term_1, term_2, Tensor_1, Tensor_2, contract, batch, keep_1, keep_2, sizes):#full_term_dict, sizes):
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
    tic = time.time()
    
    batch_idcs = ''.join(batch)
    contract_idcs = ''.join(contract)
    keep_1_idcs = ''.join(keep_1)
    keep_2_idcs = ''.join(keep_2)
    
    transpose_tuple_1 = tuple(term_1.index(i) for i in batch_idcs+''.join(keep_1)+contract_idcs)
    transpose_tuple_2 = tuple(term_2.index(i) for i in batch_idcs+contract_idcs +''.join(keep_2))

    size_batch = math.prod([sizes[i] for i in batch_idcs])
    size_sum = math.prod([sizes[i] for i in contract_idcs])
    size_rest_1 = math.prod([sizes[i] for i in keep_1_idcs])
    size_rest_2 = math.prod([sizes[i] for i in keep_2_idcs])

    Tensor_1_new = np.ascontiguousarray((np.transpose(Tensor_1, transpose_tuple_1)).reshape((size_batch, size_rest_1, size_sum )))
    Tensor_2_new = np.ascontiguousarray((np.transpose(Tensor_2, transpose_tuple_2)).reshape((size_batch, size_sum, size_rest_2)))
     
    #O_Tensor = mm.invoke_bmm(Tensor_1_new, Tensor_2_new)
    O_Tensor = mm.invoke_c_bmm(Tensor_1_new, Tensor_2_new)
    term_O = batch_idcs + keep_1_idcs + keep_2_idcs

    size_o = tuple([sizes[i] for i in term_O])
    
    O_Tensor = np.ascontiguousarray(O_Tensor.reshape(size_o))
    toc = time.time()
    print(f"bmm {toc-tic}")
    return O_Tensor, term_O