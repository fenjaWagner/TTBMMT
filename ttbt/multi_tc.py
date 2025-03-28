import numpy as np
import torch
import time
import einsum_benchmark
from ttbt import pairwise_tc

def build_sizes(term_A, term_B, shape_A, shape_B):
    """
    Builds a dictionary of sizes for each term in the tensor contraction.

    Args:
        term_A (str): Subscripts representing the indices of the first tensor.
        term_B (str): Subscripts representing the indices of the second tensor.
        shape_A (tuple): The shape of the first tensor.
        shape_B (tuple): The shape of the second tensor.

    Returns:
        dict: A dictionary mapping each index in the subscripts to its corresponding size.
    """
    sizes = {}
    sizes = dict(zip(term_A, shape_A))  # Map each term of A to its corresponding size
    sizes.update(dict(zip(term_B, shape_B)))  # Update with the terms of B and their sizes
    return sizes


def invoke_contraction_torch(A, B, mini_f_string):
    """
    Performs tensor contraction using PyTorch's einsum function.

    Args:
        A (np.ndarray): The first input tensor.
        B (np.ndarray): The second input tensor.
        mini_f_string (str): The Einstein summation string for the contraction.

    Returns:
        tuple: The resulting tensor and the time taken to perform the contraction.
    """
    A_t = torch.from_numpy(A)  # Convert NumPy arrays to torch tensors
    B_t = torch.from_numpy(B)
    tic = time.time()
    C = (torch.einsum(mini_f_string, A_t, B_t)).numpy()  # Perform contraction
    toc = time.time()
    return C, toc - tic  # Return result and time


def invoke_contraction_numpy(A, B, mini_f_string):
    """
    Performs tensor contraction using NumPy's einsum function.

    Args:
        A (np.ndarray): The first input tensor.
        B (np.ndarray): The second input tensor.
        mini_f_string (str): The Einstein summation string for the contraction.

    Returns:
        tuple: The resulting tensor and the time taken to perform the contraction.
    """
    tic = time.time()
    C = np.einsum(mini_f_string, A, B)  # Perform contraction using NumPy's einsum
    toc = time.time()
    return C, toc - tic


def invoke_contraction_np_mm(A, B, mini_f_string):
    """
    Performs tensor contraction using a custom pairwise tensor contraction method for NumPy.

    Args:
        A (np.ndarray): The first input tensor.
        B (np.ndarray): The second input tensor.
        mini_f_string (str): The Einstein summation string for the contraction.

    Returns:
        tuple: The resulting tensor and the time taken to perform the contraction.
    """
    tic = time.time()
    mini_inputs, mini_output = mini_f_string.split("->")
    mini_inputs = mini_inputs.split(",")
    term_A = mini_inputs[0]
    term_B = mini_inputs[1]
    sizes = build_sizes(term_A, term_B, A.shape, B.shape)
   
    C = pairwise_tc.pairwise_tc_np_mm(term_A, term_B, mini_output, A, B, sizes)  # Use custom pairwise contraction
    if type(C) != np.ndarray:
        C = np.array(C)  # Ensure the result is a NumPy array
    toc = time.time()
    return C, toc - tic


def invoke_contraction_custom(A, B, mini_f_string):
    """
    Performs tensor contraction using a custom pairwise tensor contraction method for custom backend.

    Args:
        A (np.ndarray): The first input tensor.
        B (np.ndarray): The second input tensor.
        mini_f_string (str): The Einstein summation string for the contraction.

    Returns:
        tuple: The resulting tensor and the time taken to perform the contraction.
    """
    tic = time.time()
    mini_inputs, mini_output = mini_f_string.split("->")
    mini_inputs = mini_inputs.split(",")
    term_A = mini_inputs[0]
    term_B = mini_inputs[1]
    sizes = build_sizes(term_A, term_B, A.shape, B.shape)
   
    C = pairwise_tc.pairwise_tc_custom(term_A, term_B, mini_output, A, B, sizes)  # Use custom pairwise contraction
    if type(C) != np.ndarray:
        C = np.array(C)  # Ensure the result is a NumPy array
    toc = time.time()
    return C, toc - tic


def prepare_contraction(mini_f_string, A, B, backend="custom"):
    """
    Prepares and invokes the appropriate tensor contraction method based on the chosen backend.

    Args:
        mini_f_string (str): The Einstein summation string for the contraction.
        A (np.ndarray): The first input tensor.
        B (np.ndarray): The second input tensor.
        backend (str): The backend to use for contraction. Options are "custom", "torch", "numpy", "np_mm". Default is "custom".

    Returns:
        tuple: The resulting tensor and the time taken for the contraction.
    """
    methods = {
        "custom": invoke_contraction_custom,
        "torch": invoke_contraction_torch,
        "numpy": invoke_contraction_numpy,
        "np_mm": invoke_contraction_np_mm,
    }
    method = methods[backend]
    try:
        C, time_fragment = method(A, B, mini_f_string)  # Call the appropriate contraction method
    except Exception as e:
        print(f"Error in {backend}: {e}")
        C, time_fragment = None, None  # Return None if there was an error
    return C, time_fragment


def multi_tc(path, tensors, format_string, backend="custom"):
    """
    Perform multiple tensor contractions based on the given contraction path.

    Args:
        path (str): The path (not in ssa format!) defining the sequence of contractions.
        tensors (list): A list of tensors to be used in the contractions.
        format_string (str): The Einstein summation format string.
        backend (str): The backend to use for tensor contractions. Default is "custom".

    Returns:
        tuple: The final contracted tensor, the total time taken for all contractions, and the time spent on individual fragments.
    """
    ssa_path = einsum_benchmark.meta.runtime.to_ssa_path(path)  # Convert path to SSA format
    format_string = format_string.replace(" ", "")  # Clean up the format string
    annotated_path = einsum_benchmark.meta.runtime.to_annotated_ssa_path(format_string, ssa_path, True)  # Annotate the SSA path
    length = len(tensors)
    tic = time.time()
    time_fragment = 0
    
    # Iterate over the annotated path to perform each contraction step
    for t_tuple in annotated_path:
        first = t_tuple[0]
        second = t_tuple[1]
        mini_f_string = t_tuple[2]
        A = tensors[first]
        B = tensors[second]   
        
        # Invoke the appropriate contraction method
        C, time_fragment_tmp = prepare_contraction(mini_f_string, A, B, backend)
        
        # Delete used tensors if they are not in the original tensor list
        if first >= length:
            A = None #np.empty((1,), dtype=np.float32) 
        if second >= length:
            B = None #np.empty((1,), dtype=np.float32) 
        
        time_fragment += time_fragment_tmp  # Add the time for this fragment
        tensors.append(C)  # Append the result to the tensors list
        
    toc = time.time()
    tensors = tensors[:length]  # Trim the tensors list to its original size
    
    return C, toc - tic, time_fragment  # Return final result, total time, and the time each backend used for the contractions
