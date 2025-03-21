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
    """
    Invokes a custom batched matrix multiplication (BMM) implemented in C++.

    Args:
        Tensor_1 (torch.Tensor): The first input tensor for the batched matrix multiplication.
        Tensor_2 (torch.Tensor): The second input tensor for the batched matrix multiplication.

    Returns:
        torch.Tensor: The result of the custom C++ batched matrix multiplication.
    """
    return wrapper.call_cpp_bmm(Tensor_1, Tensor_2)


def invoke_np_mm(Tensor_1, Tensor_2):
    """
    Performs standard matrix multiplication using NumPy or PyTorch.

    Args:
        Tensor_1 (Tensor or ndarray): The first input tensor or array.
        Tensor_2 (Tensor or ndarray): The second input tensor or array.

    Returns:
        The result of the matrix multiplication (Tensor or ndarray), depending on input types.
    """
    return Tensor_1 @ Tensor_2



def pairwise_tc(term_1, term_2, term_O, Tensor_1, Tensor_2, sizes, backend = "custom"):
    """
    Performs a pairwise tensor contraction by reducing the problem to a batched matrix multiplication (BMM).

    Args:
        term_1 (str): Subscripts representing the indices of the first tensor.
        term_2 (str): Subscripts representing the indices of the second tensor.
        term_O (str): Subscripts representing the indices of the output tensor.
        Tensor_1 (np.ndarray or torch.Tensor): The first input tensor.
        Tensor_2 (np.ndarray or torch.Tensor): The second input tensor.
        sizes (dict): A dictionary mapping each index to its dimension size.
        backend (str, optional): Backend to use for BMM. Options: "custom" or "np_mm". Default is "custom".

    Returns:
        np.ndarray or torch.Tensor: The resulting contracted tensor.
    """

    # Map backend string to function (custom C++ BMM or NumPy matrix multiplication)
    backend_dict = {"custom": invoke_custom_bmm,
                    "np_mm": invoke_np_mm}
    tc_method = backend_dict[backend]

    # Identify which indices are batch, contract, or keep indices
    batch_idcs, contract_idcs, keep_1_idcs, keep_2_idcs = prepare_indices(term_1, term_2, term_O)
    
    # Ensure both tensors are of the same result type
    result_type = np.result_type(Tensor_1)
    if np.result_type(Tensor_2) != result_type:
        Tensor_2.astype(result_type)

    # Compute the product of sizes for batch, contraction, and remaining dimensions
    size_batch = math.prod([sizes[i] for i in batch_idcs])
    size_sum = math.prod([sizes[i] for i in contract_idcs])
    size_rest_1 = math.prod([sizes[i] for i in keep_1_idcs])
    size_rest_2 = math.prod([sizes[i] for i in keep_2_idcs])

    # Reorder and reshape the tensors according to batch, keep, and contract indices
    if term_1:
        Tensor_1 = np.einsum(term_1 +'->'+batch_idcs+keep_1_idcs+contract_idcs, Tensor_1)
    if term_2:
        Tensor_2 = np.einsum(term_2 +'->'+batch_idcs+contract_idcs+keep_2_idcs, Tensor_2)

    # Reshape tensors into shapes compatible with BMM: 
    # Tensor_1: (batch, M, K), Tensor_2: (batch, K, N)
    Tensor_1 = np.ascontiguousarray((Tensor_1).reshape((size_batch, size_rest_1, size_sum )))
    Tensor_2 = np.ascontiguousarray((Tensor_2).reshape((size_batch, size_sum, size_rest_2)))

    # Perform the batched matrix multiplication using the selected backend
    O_Tensor = tc_method(Tensor_1, Tensor_2)

    term_O_tmp = batch_idcs + keep_1_idcs + keep_2_idcs
    

    size_o = tuple([sizes[i] for i in term_O_tmp])
    
    # Reconstruct the output tensor by reshaping and permuting back to the desired output ordering
    O_Tensor = np.ascontiguousarray(O_Tensor.reshape(size_o))
    if term_O != term_O_tmp:
        O_Tensor = np.einsum(term_O_tmp+"->"+term_O, O_Tensor)

    return O_Tensor

def pairwise_tc_custom(term_1, term_2, term_O, Tensor_1, Tensor_2, sizes):
    """
    Shortcut function to call pairwise_tc with the custom C++ backend.
    """
    return pairwise_tc(term_1, term_2, term_O, Tensor_1, Tensor_2, sizes, "custom")


def pairwise_tc_np_mm(term_1, term_2, term_O, Tensor_1, Tensor_2, sizes):
    """
    Shortcut function to call pairwise_tc using standard NumPy matrix multiplication as the backend.
    """
    return pairwise_tc(term_1, term_2, term_O, Tensor_1, Tensor_2, sizes, "np_mm")