import numpy as np

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
    """Creates a dict with all indices and their position in the term.

    Args:
        term (str): Term

    Returns:
        dict: Index dict
    """
    index_dict = {}
    for i in range(len(term)):
        index_dict[term[i]] = i
    return index_dict


def tuple_to_list(tup):
    li = []
    for i in tup:
        li.append(i)
    return li

def list_to_string(li):
    st = ''
    for i in li:
        st += str(i)
    return st


def transpose_tuple(term_1, term_transposed) -> tuple:
    index_dict_1 = create_index_dict(term_1)
    transpose_list = []
    for i in term_transposed:
        transpose_list.append(index_dict_1[i])
    
    return tuple(transpose_list)


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



# single_trace

def sum_shape(shape):
    shape_sum = [0,]*len(shape)
    prod = 1
    for i in range(len(shape)-1, -1, -1):
        shape_sum[i] = prod
        prod *= shape[i]
    return shape_sum

def calc_new_length(shape):
    prod = 1
    for i in shape:
        prod *= i
    return prod

def create_iterator(shape):
    iterator = np.ndindex(tuple(shape))
    return iterator

def calc_positions(index, sum_new_shape, sum_old_shape, new_term,term_dict):
    
    pos_A_new = 0
    pos_A_old = 0
    for i in range(len(index)):
        pos_A_new += index[i]*sum_new_shape[i]
        for j in term_dict[new_term[i]]:
            pos_A_old += index[i] * sum_old_shape[j]
    return pos_A_old, pos_A_new