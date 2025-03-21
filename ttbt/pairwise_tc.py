import numpy as np
import math
import time
from ttbt import wrapper

def prepare_indices(term_A, term_B,  mini_output):
    """Prepares index sets for batched, contracted, and kept dimensions based on the input terms.

Args:
    term_A (iterable): Indices associated with tensor A.
    term_B (iterable): Indices associated with tensor B.
    mini_output (iterable): Indices expected to appear in the output.

Returns:
    tuple: A tuple of four sets:
        - batch (set): Indices present in all three sets (term_A, term_B, mini_output),
                        typically used for batched operations.
        - contract (set): Indices present in both term_A and term_B, but not in mini_output,
                            typically contracted out during operations.
        - keep_A (set): Indices in both term_A and mini_output but not in term_B,
                        these are retained on the A side.
        - keep_B (set): Indices in both term_B and mini_output but not in term_A,
                        these are retained on the B side.

Notes:
    - If the computed keep_A or keep_B sets are empty (and no batch or contract indices are found),
        they default to the entire term_A or term_B sets respectively, to avoid empty tensor operations.
"""
    set_A = set(term_A)
    set_B = set(term_B)
    output_set = set(mini_output)    
    
    batch = set_A & set_B & output_set
    contract = (set_A & set_B) - output_set
    keep_A = (set_A & output_set) - set_B
    keep_B = (set_B & output_set) - set_A

    # avoid empty tensors
    if not (batch or contract or keep_A):
        keep_A = set_A
    if not (batch or contract or keep_B):
        keep_B = set_B

    return [''.join(set_l) for set_l in [batch, contract, keep_A, keep_B]]

def invoke_custom_bmm(Tensor_1, Tensor_2):
    return wrapper.call_cpp_bmm(Tensor_1, Tensor_2)

def invoke_np_mm(Tensor_1, Tensor_2):
    return Tensor_1 @ Tensor_2


def pairwise_tc(term_1, term_2, term_O, Tensor_1, Tensor_2, sizes, backend = "custom"):
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
    backend_dict = {"custom": invoke_custom_bmm,
                    "np_mm": invoke_np_mm}
    tc_method = backend_dict[backend]
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

    O_Tensor = tc_method(Tensor_1, Tensor_2)

    term_O_tmp = batch_idcs + keep_1_idcs + keep_2_idcs
    

    size_o = tuple([sizes[i] for i in term_O_tmp])
    
    O_Tensor = np.ascontiguousarray(O_Tensor.reshape(size_o))
    if term_O != term_O_tmp:
        O_Tensor = np.einsum(term_O_tmp+"->"+term_O, O_Tensor)

    return O_Tensor

def pairwise_tc_custom(term_1, term_2, term_O, Tensor_1, Tensor_2, sizes):
    return pairwise_tc(term_1, term_2, term_O, Tensor_1, Tensor_2, sizes, "custom")


def pairwise_tc_np_mm(term_1, term_2, term_O, Tensor_1, Tensor_2, sizes):
    return pairwise_tc(term_1, term_2, term_O, Tensor_1, Tensor_2, sizes, "np_mm")