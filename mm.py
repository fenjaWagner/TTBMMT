import numpy as np
import re
import torch

# TODO: ersetze shape durch sizes dict
def create_iterator(term_dict, shape):
    """Creates an iterator that iterates over all the different indices in the term. 

    Args:
        term_dict (dict): Dictionary that holds the position of an idex in the term.
        shape (tuple): Shape of the tensor. 

    Returns:
        iterator: Iterator
        shape_l: Shape of the new tensor.
        index_l: List of the new term for the processed te
    """
    shape_l = []
    index_l = []
    for k in term_dict:
        shape_l.append(shape[term_dict[k][0]])
        index_l.append(k)
    shape_tuple = tuple(shape_l)
    iterator = np.ndindex(shape_tuple)
    return iterator, shape_l, index_l

def calculate_shape_prod(shape):
    shape_sum = [0,]*len(shape)
    prod = 1
    for i in range(len(shape)-1, -1, -1):
        shape_sum[i] = prod
        prod *= shape[i]
    return shape_sum


def single_trace(term: str, A: np.array, shape: tuple):
    shape_sum = calculate_shape_prod(shape)   
    #term_list = [x for x in term]

    # create dict that holds the positions of the indices
    term_dict = {}
    for i in range(len(term)):
        if term[i] not in term_dict:
            term_dict[term[i]] = [i]
        else: 
            term_dict[term[i]].append(i)
    

    iterator, shape_new, index_new = create_iterator(term_dict, shape)

    new_shape_sum = calculate_shape_prod(shape_new)

    prod = 1
    for i in shape_new:
        prod *= i

    A_new = np.zeros(prod)

    for index in iterator:
        index_A = 0
        index_A_new = 0
        for i in range(len(index)):
            index_A_new += index[i]*new_shape_sum[i]
            variable = index_new[i]
            for l in term_dict[variable]:
                index_A += index[i]*shape_sum[l]
        
        A_new[index_A_new] += A[index_A]
    
    A_new = np.ascontiguousarray(A_new.reshape(tuple(shape_new)))
    return A_new, str(index_new)  



def matmul_bmm(A: np.array, B: np.array, C: np.array, size_A: tuple, size_B: tuple, batch_idx: int) -> np.array:
    round_index_c = batch_idx * size_A[1] * size_B[2]
    round_index_a = batch_idx * size_A[1] * size_A[2]
    round_index_b = batch_idx * size_B[1] * size_B[2]
    
    for i in range(size_A[1]):
        row_offset_A = round_index_a + i * size_A[2]
        row_offset_B = round_index_c + i * size_B[2]

        for j in range(size_A[2]):
            element_A = A[row_offset_A + j]
            col_offset_B = round_index_b + j * size_B[2]

            for k in range(size_B[2]):
                C[row_offset_B + k] += element_A * B[col_offset_B + k]

def bmm(A: np.array, B: np.array, C: np.array, shape_a, shape_b):
    if shape_a[0] is not shape_b[0]:
        raise Exception("Dimension of Tensors don't match.")
    #prepare_matmul(A, B, C)
    
    for batch_idx in range(shape_a[0]):
        matmul_bmm(A, B, C, shape_a, shape_b, batch_idx)
    return C


def invoke_bmm(A: np.array, B: np.array) -> np.array:
    shapeA = A.shape
    shapeB = B.shape
    C = np.zeros((A.shape[0]* A.shape[1]* B.shape[2]))
    A = np.ascontiguousarray(A.reshape(-1))
    B = np.ascontiguousarray(B.reshape(-1))

    bmm(A,B, C, shapeA, shapeB)

    C = C.reshape((shapeA[0], shapeA[1], shapeB[2]))
    return C
    
    

def create_set(term: str) -> set:
    """Creates a set of all indices that are in the term.

    Args:
        term (str): term consisiting of the indices.

    Returns:
        set: Set that contains every index.
    """
    set_term = set()
    for i in term:
        set_term.add(i)
    return set_term

def create_index_dict(term: str) -> dict:
    index_dict = {}
    for i in range(len(term)):
        index_dict[term[i]] = i
    return index_dict

def create_transpose_tuple_term(term:str, batch_dim: list, sum_dim: list, index_sum, sizes: dict) -> tuple:
    """Generates the transpose tuple for the given parameters for the tensor.

    Args:
        term (str): The term with all the indices of the tensor.
        batch_dim (str): Name of the batch dimension.
        sum_dim (str): Name of the dimension that is summed over.
        index_sum (int): Position of the summation index in the transposed tensor.
        sizes (dict): Dictionary that holds the sizes for the indices in the terms.

    Returns:
        tuple(transpose_l): Tuple that indicates how the tensor needs to be transposed.
        tuple(index_l): Contains the indices in the order after transposing.
        size_prod(int): Size of the dimension between sum and batch dimension.
    """
    term_length = len(term)
    batch_length = len(batch_dim)
    sum_length = len(sum_dim)

    index_dict = create_index_dict(term)
    index_l = [0,]*term_length
    transpose_l = [0,]* term_length

    
    # insert batch dimension into transpose 
    index_l[0: batch_length] = [x for x in batch_dim]
    transpose_l[0:batch_length] = [index_dict.pop(x) for x in batch_dim]

    #index_l[0] = batch_dim
    #transpose_l[0] = index_dict.pop(batch_dim)

    if index_sum == 1:
        index_l[batch_length: batch_length+ sum_length] = [x for x in sum_dim]
        transpose_l[batch_length: batch_length+ sum_length] = [index_dict.pop(x) for x in sum_dim]


    #index_l[index_sum] = sum_dim
    #transpose_l[index_sum] = index_dict.pop(sum_dim)


    # Calculate size of the dimensions except batch and sum dimension.
    size = [sizes[k] for k in index_dict]
    size_prod = 1
    for i in size:
        size_prod *= i


    if term_length > 2:
        if index_sum == 1:
            index_l[2:] = [k for k in index_dict]
            transpose_l[2:] = [index_dict[k] for k in index_dict]
        else:
            index_l[1:-1] = [k for k in index_dict]
            transpose_l[1:-1] = [index_dict[k] for k in index_dict]
    return tuple(transpose_l), tuple(index_l), size_prod



def find_mapping(term1: str, term2: str, A: np.array, B: np.array, term_output: str, sizes: dict):
    size_o = []
    for i in term_output:
        size_o.append(sizes[i])

    set_term2 = create_set(term2)
    set_output = create_set(term_output)

    batch_dim = []
    sum_dim = []

    for i in term1:
        if i in set_term2:
            if i in set_output:
                batch_dim.append(i)
            else: 
                sum_dim.append(i)
    
    term_1_transpose, term_1_index, size_prod_1 = create_transpose_tuple_term(term1, batch_dim, sum_dim, -1, sizes)
    term_2_transpose, term_2_index, size_prod_2 = create_transpose_tuple_term(term2, batch_dim, sum_dim, 1, sizes)

    shape_1 = (sizes[batch_dim], size_prod_1, sizes[sum_dim])
    shape_2 = (sizes[batch_dim], sizes[sum_dim], size_prod_2)

    A_new = np.ascontiguousarray((np.transpose(A, term_1_transpose)).reshape(shape_1))
    B_new = np.ascontiguousarray((np.transpose(B, term_2_transpose)).reshape(shape_2))

    O = invoke_bmm(A_new, B_new)
    O = np.ascontiguousarray(O.reshape(tuple(size_o)))

    return O




def read_input_string(string, arrays):

    input_terms, output = string.split("->")
    input = input_terms.split(",")
    size = {}

    for term, array in zip(input, arrays):
        for k, d in zip(term, array.shape):
            size[k] = d


    

def test_bmm():
    A = np.random.rand(10, 15, 11)
    B = np.random.rand(10, 11, 24)

    A_t = torch.from_numpy(A)
    B_t = torch.from_numpy(B)
    C_torch = torch.bmm(A_t, B_t)

    
    C =  invoke_bmm(A, B)
    #A.reshape(-1)
    #B.reshape(-1)
    
    C_t = torch.from_numpy(C)
    print((C_torch - C_t).sum())

def test_manage_input_strings():
    string = "ijk, kjl  -> zs"
    read_input_string(string)



def test_trace():
    A = np.random.rand(5,5,4,3,5)
    print(A)
    #A = np.array([[1,2,3], [3,4,5], [5,6,7]])
    string = "iikji"
    At = torch.from_numpy(A)
    B = A.reshape(-1)
    D = torch.einsum("iikji->ikj", At)
    print(D)
    C, index = single_trace(string, B, A.shape)
    Ct = torch.from_numpy(C)
    print((Ct-D).sum())

def test_find_mapping():
    A = np.random.rand(3,4,6,5)
    B = np.random.rand(3,5,6,2)

    At = torch.from_numpy(A)
    Bt = torch.from_numpy(B)
    

    T = torch.einsum("ijmk, ikln -> ijmln", At,Bt)
    str_A = "ijmk"
    str_B = "ikln"
    str_O = "ijmln"
    sizes = {"i": 3,
             "j": 4, 
             "k": 5,
             "l": 6,
             "m": 6,
             "n": 2}

    U = find_mapping(str_A, str_B, A, B, str_O,sizes)
    Ut = torch.from_numpy(U)

    
    print((T-Ut).sum())


test_trace()







