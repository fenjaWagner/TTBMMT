import torch 
import numpy as np
import re
import mm 


def create_index_dict(term: str) -> dict:
    """Creates a dict with all indices in the term.

    Args:
        term (str): Term

    Returns:
        dict: Index dict
    """
    index_dict = {}
    for i in range(len(term)):
        index_dict[term[i]] = i
    return index_dict

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

def transpose_tuple(term_1, term_transposed) -> tuple:
    index_dict_1 = create_index_dict(term_1)
    transpose_list = []
    for i in term_transposed:
        transpose_list.append(index_dict_1[i])
    
    return tuple(transpose_list)

def find_batch_sum_terms(term_1, term_2, term_output):
    """Retreives all batch and sum dimensions from the given term.

    Args:
        term_1 (str): _description_
        term_2 (str): _description_
        term_output (str): _description_

    Returns:
        batch_terms (list), sum_terms (list): Lists containing the respective indices.
    """
    set_term_2 = create_set(term_2)
    set_output = create_set(term_output)

    batch_terms = []
    sum_terms = []

    for i in term_1:
        if i in set_term_2:
            if i in set_output:
                batch_terms.append(i)
            else: 
                sum_terms.append(i)

    return batch_terms, sum_terms


def generate_transposed_term(term, batch_terms, sum_terms, case):
    """Generates the transposed term and the tuple to transpose the tensor considering the batch and sum terms. 

    Args:
        term (str): _description_
        batch_terms (list): _description_
        sum_terms (list): _description_
        case (int): _description_

    Returns:
        _type_: _description_
    """
    term_length = len(term)
    batch_length = len(batch_terms)
    sum_length = len(sum_terms)
    index_dict = create_index_dict(term)

    #create sublists for the different subterms (batch dimension, summation dimension, rest)
    batch_index_list = [index_dict.pop(x) for x in batch_terms]
    sum_index_list = [index_dict.pop(x) for x in sum_terms]
    rest_terms = [x for x in index_dict]
    rest_index_list = [index_dict[x] for x in rest_terms]
    
    if case == "first":
        term_transpose = batch_index_list + rest_index_list + sum_index_list
        new_term_list = batch_terms + rest_terms + sum_terms
    
    elif case == "second":
        term_transpose = batch_index_list  + sum_index_list + rest_index_list
        new_term_list = batch_terms  + sum_terms + rest_terms

    new_term = ""
    for i in new_term_list:
        new_term += i

    return tuple(term_transpose), new_term, rest_terms


def calculate_size(term, sizes):
    """Calculates the size of an tensor given its index term. 

    Args:
        term (str): _description_
        sizes (dict): _description_

    Returns:
        int: _description_
    """
    size = 1
    for i in term:
        size *= sizes[i]
    return size

def sum_str(list):
    new_str = ""
    for s in list:
        new_str += str(s)
    return new_str
    

def normal_mapping(term_1, term_2, Tensor_1, Tensor_2, term_output, sizes):
    """Maps the contraction of two given tensors to the bmm.

    Args:
        term_1 (str): _description_
        term_2 (str): _description_
        Tensor_1 (np.array): _description_
        Tensor_2 (np.array): _description_
        term_output (str): _description_
        sizes (dict): _description_

    Returns:
        _type_: _description_
    """
    
    batch_terms, sum_terms = find_batch_sum_terms(term_1, term_2, term_output)
    transpose_tuple_1, new_term_1, rest_terms_1 = generate_transposed_term(term_1, batch_terms, sum_terms, "first")
    transpose_tuple_2, new_term_2, rest_terms_2 = generate_transposed_term(term_2, batch_terms, sum_terms, "second")

    size_batch = calculate_size(batch_terms, sizes)
    size_sum = calculate_size(sum_terms, sizes)
    size_rest_1 = calculate_size(rest_terms_1, sizes)
    size_rest_2 = calculate_size(rest_terms_2, sizes)

    Tensor_1_new = np.ascontiguousarray((np.transpose(Tensor_1, transpose_tuple_1)).reshape((size_batch, size_rest_1, size_sum )))
    tensor_2_new = np.ascontiguousarray((np.transpose(Tensor_2, transpose_tuple_2)).reshape((size_batch, size_sum, size_rest_2)))

    O_Tensor = mm.invoke_bmm(Tensor_1_new, tensor_2_new)
    term_O = sum_str(batch_terms) + sum_str(rest_terms_1) + sum_str(rest_terms_2)

    size_o = []
    for i in term_O:
        size_o.append(sizes[i])

    # TODO: Zurücktransponieren
    O_Tensor = np.ascontiguousarray(O_Tensor.reshape(tuple(size_o)))

    return O_Tensor, term_O





def test_mapping():
    A = np.random.rand(3,4,6,5)
    B = np.random.rand(3,5,6,2)

    At = torch.from_numpy(A)
    Bt = torch.from_numpy(B)
    

    str_A = "ijmk"
    str_B = "ikln"
    str_O = "ijmln"
    sizes = {"i": 3,
             "j": 4, 
             "k": 5,
             "l": 6,
             "m": 6,
             "n": 2}

    U, term = normal_mapping(str_A, str_B, A, B, str_O,sizes)
    print("term: ", term)
    T = torch.einsum("ijmk, ikln ->" + term, At,Bt)
    Ut = torch.from_numpy(U)

    
    print((T-Ut).sum())

    
    
test_mapping()
    
