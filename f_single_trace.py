import numpy as np
import useful_funcs
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


def single_trace(term, A, sizes):
    term_dict = create_dict(term)
    new_term, new_shape = generate_new_term(term_dict, sizes)
    iterator = useful_funcs.create_iterator(new_shape)
    sum_old_shape = useful_funcs.sum_shape(A.shape)
    sum_new_shape = useful_funcs.sum_shape(new_shape)

    prod = useful_funcs.calc_new_length(new_shape)
    A_new = np.zeros(prod)
    A = A.reshape(-1)
    
    for index in iterator:
        pos_A_old, pos_A_new = useful_funcs.calc_positions(index, sum_new_shape, sum_old_shape,  new_term,term_dict)
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

    #B = A.reshape(-1)
    C, new_term = single_trace(string, A, sizes)
    print("new_term: ", new_term)
    D = np.einsum("iikjij->"+new_term, A)
    
    
    print((C-D).sum())

#test_trace()

