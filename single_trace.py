import torch 
import numpy as np
import re
import mm 


def create_dict(term):
    term_dict = {}
    for i in range(len(term)):
        if term[i] not in term_dict:
            term_dict[term[i]] = [i]
        else: 
            term_dict[term[i]].append(i)
    return term_dict

def generate_new_term(term_dict, sizes):
    new_term = ""
    new_shape = []
    for k in term_dict:
        new_term += str(k)
        new_shape.append(sizes[k])
    return new_term, new_shape

def create_iterator(shape):
    iterator = np.ndindex(tuple(shape))
    return iterator

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

def single_trace(term, A, sizes):
    term_dict = create_dict(term)
    new_term, new_shape = generate_new_term(term_dict, sizes)
    iterator = create_iterator(new_shape)
    sum_old_shape = sum_shape(A.shape)
    sum_new_shape = sum_shape(new_shape)

    prod = calc_new_length(new_shape)
    A_new = np.zeros(prod)
    A = A.reshape(-1)

    for index in iterator:
        pos_A_new = 0
        pos_A_old = 0
        for i in range(len(index)):
            pos_A_new += index[i]*sum_new_shape[i]
            for j in term_dict[new_term[i]]:
                pos_A_old += index[i] * sum_old_shape[j]
            A_new[pos_A_new] = A[pos_A_old]
            
    A_new = A_new.reshape(tuple(new_shape))
    return A_new, new_term



def test_trace():
    A = np.random.rand(5,5,4,3,5,3)
    #print(A)
    #A = np.array([[1,2,3], [3,4,5], [5,6,7]])
    string = "iikjij"

    sizes = {}
    for i in range(len(string)):
        sizes[string[i]] = A.shape[i]

    At = torch.from_numpy(A)
    #B = A.reshape(-1)
    C, new_term = single_trace(string, A, sizes)
    print("new_term: ", new_term)
    D = torch.einsum("iikjij->"+new_term, At)
    
    Ct = torch.from_numpy(C)
    
    print((Ct-D).sum())

test_trace()

