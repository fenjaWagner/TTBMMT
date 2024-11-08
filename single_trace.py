import torch 
import numpy as np
import re
import mm 

def find_double_indices(term):
    dim_set = set()
    single_indices = {}
    double_indices = {}

    for i in range(len(term)):
        dim = term[i]
        if dim in dim_set:
            if dim in single_indices:
                i_list = single_indices.pop(dim)
                double_indices[dim] = i_list
            double_indices[dim].append(i)
        else:
            dim_set.add(dim)
            single_indices[dim] = [i]

    return single_indices, double_indices


def add_lists(i_dict):
    new_list = []
    for k in i_dict:
        new_list += i_dict[k]
    return new_list

def add_terms(term_dict, factor = 0):
    new_term = ""
    for k in term_dict:
        if factor == 0:
            new_term += str(k)
        else:
            new_term += str(k)* len(term_dict[k])
    return new_term

def generate_new_shape(new_term, sizes):
    new_shape = []
    for i in new_term:
        new_shape.append(sizes[i])
    return new_shape

        

def create_transpose_tuple(single_indices, double_indices, sizes):
    single_transpose = add_lists(single_indices)
    double_transpose = add_lists(double_indices)
    single_term = add_terms(single_indices)
    double_term = add_terms(double_indices)
    double_term_complete = add_terms(double_indices, factor = 1)

    new_term_short = single_term + double_term
    new_term_complete = single_term + double_term_complete

    transpose_list = tuple(single_transpose + double_transpose)
    
    return transpose_list, new_term_short, new_term_complete



def create_iterator(new_shape):#term_dict, shape):
    """Creates an iterator that iterates over all the different indices in the term. 

    Args:
        new_shape (tuple): New shape of the tensor. 

    Returns:
        iterator: Iterator
    """
    iterator = np.ndindex(new_shape)
    return iterator

def calculate_shape_prod(shape):
    shape_sum = [0,]*len(shape)
    prod = 1
    for i in range(len(shape)-1, -1, -1):
        shape_sum[i] = prod
        prod *= shape[i]
    return shape_sum


def calc_single_trace(term: str, A: np.array, sizes: dict):
      
    single_indices, double_indices = find_double_indices(term)
    transpose_tuple, new_term_short, new_term_complete = create_transpose_tuple(single_indices, double_indices, sizes)
    
    print("single:", single_indices)
    print("double:", double_indices)
    if single_indices:
        print("if")
        joint_indices = single_indices.copy()
        joint_indices.update(double_indices)
        print("joint: ", joint_indices)
    else: 
        joint_indices = double_indices
    

    new_shape_short = generate_new_shape(new_term_short, sizes)
    iterator = create_iterator(tuple(new_shape_short))
    new_shape_complete = generate_new_shape(new_term_complete, sizes)

    A = np.ascontiguousarray(np.transpose(A, transpose_tuple))
    A = np.ascontiguousarray(A.reshape(-1))
    
    shape_sum_short = calculate_shape_prod(new_shape_short)
    shape_sum_complete = calculate_shape_prod(new_shape_complete)

    prod = 1
    for i in new_shape_short:
        prod *= i
    A_new = np.zeros(prod)
    A = A.reshape(-1)

    for index in iterator:   
        pos_A_new = 0
        pos_A = 0         
        for i in range(len(index)):
            pos_A_new += index[i]*shape_sum_short[i]
            variable = new_term_short[i]
            for l in joint_indices[variable]:
                pos_A += index[i]*shape_sum_complete[l]
            
        A_new[pos_A_new] += A[pos_A]
    
    A_new = np.ascontiguousarray(A_new.reshape(new_shape_short))
    return A_new, new_term_short 



def test_trace():
    A = np.random.rand(5,5,4,3,5)
    #print(A)
    #A = np.array([[1,2,3], [3,4,5], [5,6,7]])
    string = "iikji"

    sizes = {}
    for i in range(len(string)):
        sizes[string[i]] = A.shape[i]

    At = torch.from_numpy(A)
    #B = A.reshape(-1)
    C, index = calc_single_trace(string, A, sizes)
    D = torch.einsum("iikji->"+index, At)
    
    Ct = torch.from_numpy(C)
    
    print((Ct-D).sum())

test_trace()