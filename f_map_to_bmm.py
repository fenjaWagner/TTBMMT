#import torch 
import numpy as np
import mm 
import useful_funcs
    
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
    index_dict = useful_funcs.create_index_dict(term)

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

    elif case == "no_sum":
        term_transpose = batch_index_list + rest_index_list
        new_term_list = batch_terms + rest_terms

    new_term = useful_funcs.list_to_string(new_term_list)

    return tuple(term_transpose), new_term, rest_terms


def map_to_bmm(term_1, term_2, Tensor_1, Tensor_2, contract, batch, sizes):#full_term_dict, sizes):
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
    
    #batch_terms = useful_funcs.generate_list_from_set(full_term_dict["batch"])
    #sum_terms = useful_funcs.generate_list_from_set(full_term_dict["contract"])
    batch_terms = useful_funcs.generate_list_from_set(batch)
    sum_terms = useful_funcs.generate_list_from_set(contract)



    transpose_tuple_1, new_term_1, rest_terms_1 = generate_transposed_term(term_1, batch_terms, sum_terms, "first")
    transpose_tuple_2, new_term_2, rest_terms_2 = generate_transposed_term(term_2, batch_terms, sum_terms, "second")

    size_batch = useful_funcs.calculate_size(batch_terms, sizes)
    size_sum = useful_funcs.calculate_size(sum_terms, sizes)
    size_rest_1 = useful_funcs.calculate_size(rest_terms_1, sizes)
    size_rest_2 = useful_funcs.calculate_size(rest_terms_2, sizes)

    Tensor_1_new = np.ascontiguousarray((np.transpose(Tensor_1, transpose_tuple_1)).reshape((size_batch, size_rest_1, size_sum )))
    Tensor_2_new = np.ascontiguousarray((np.transpose(Tensor_2, transpose_tuple_2)).reshape((size_batch, size_sum, size_rest_2)))
     
    #O_Tensor = mm.invoke_bmm(Tensor_1_new, Tensor_2_new)
    O_Tensor = mm.invoke_c_bmm(Tensor_1_new, Tensor_2_new)
    term_O = useful_funcs.sum_str(batch_terms) + useful_funcs.sum_str(rest_terms_1) + useful_funcs.sum_str(rest_terms_2)

    size_o = []
    for i in term_O:
        size_o.append(sizes[i])

    # TODO: Zurücktransponieren
    O_Tensor = np.ascontiguousarray(O_Tensor.reshape(tuple(size_o)))

    return O_Tensor, term_O



def test_mapping_case_normal():
    A = np.random.rand(3,4,6,5)
    B = np.random.rand(3,5,6,2)

    str_A = "ijmk"
    str_B = "ikln"
    str_O = "ijmln"
    sizes = {"i": 3,
             "j": 4, 
             "k": 5,
             "l": 6,
             "m": 6,
             "n": 2}
    batch = {"i"}
    contract = {"k"}
    full_term_dict = {"batch": {"i"},
                      "contract": {"k"}}

    U, term = map_to_bmm(str_A, str_B, A, B, contract, batch,sizes)
    print("term: ", term)
    T = np.einsum("ijmk, ikln ->" + term, A,B)
    
    
    print((T-U).sum())

def test_mapping_no_sum():
    i = [[1.1,1,1,1],[2,2,2,2]]
    j = [[8,9,10], [3,4,5]]
    k = [11,11,11]

    I = np.array(i, dtype=int)
    J = np.array(j, dtype = int)
    contract = set()
    batch =  {"l"}
    full_term_dict = {"batch": {"l"},
                      "contract": set()}
    O, term = map_to_bmm("li", "lj", I, J, contract, batch, {"i": 4, "l": 2, "j": 3})
    
    C = np.einsum("li,lj ->"+ term, I, J)

    print((C -O).sum())


def test_mapping_no_batch():
    A = np.random.rand(3,4,6,5)
    B = np.random.rand(3,5,6,2)

    
    str_A = "ijmk"
    str_B = "zkln"
    str_O = "izjmln"
    sizes = {"i": 3,
             "j": 4, 
             "k": 5,
             "l": 6,
             "m": 6,
             "n": 2,
             "z": 3}
    full_term_dict = {"batch": set(),
                      "contract": {"k"}}

    U, term = map_to_bmm(str_A, str_B, A, B, {"k"}, set(),sizes)
    print("term: ", term)
    T = np.einsum("ijmk, zkln ->" + term, A,B)
    
    print((T-U).sum())

def test_mapping_no_batch_no_sum():
    A = np.random.rand(3,4,6,5)
    B = np.random.rand(3,5,6,2)


    str_A = "ijmk"
    str_B = "z4ln"
    str_O = "iz4jmln"
    sizes = {"i": 3,
             "j": 4, 
             "k": 5,
             "4": 5,
             "l": 6,
             "m": 6,
             "n": 2,
             "z": 3}
    full_term_dict = {"batch": set(),
                      "contract": set()}

    U, term = map_to_bmm(str_A, str_B, A, B, set(), set(),sizes)
    print("term: ", term)
    T = np.einsum("ijmk, zyln ->ijmkzyln", A,B)
   
    
    print((T-U).sum())
    

def test():
    print("test_no_batch_no_sum")    
    test_mapping_no_batch_no_sum()
    print("test_no_sum")
    test_mapping_no_sum()
    print("test_no_batch")
    test_mapping_no_batch()
    print("complete")
    test_mapping_case_normal()


def test_without_torch():
    A = np.random.rand(3,4,6,5)
    B = np.random.rand(3,5,6,2)

 

    str_A = "ijmk"
    str_B = "ikln"
    str_O = "ijmln"
    sizes = {"i": 3,
             "j": 4, 
             "k": 5,
             "l": 6,
             "m": 6,
             "n": 2}
    full_term_dict = {"batch": {"i"},
                      "contract": {"k"}}

    U, term = map_to_bmm(str_A, str_B, A, B, full_term_dict,sizes)
    print("term: ", term)



#test()