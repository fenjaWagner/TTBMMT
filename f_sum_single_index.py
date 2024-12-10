import torch 
import numpy as np
import  useful_funcs


def build_transpose_new_term(term, single, term_dict):
    new_term = ""
    transpose_term = ""
    single_term = ""
    for i in term:
        if i not in single:
            transpose_term += i*term_dict[i]
            new_term += i
    for i in single:
        transpose_term += i
        single_term += i
    return transpose_term, new_term, single_term

def transpose_tensor(term, transpose_term, tensor: np.array):
    transpose_tuple = useful_funcs.transpose_tuple(term, transpose_term)
    tensor = np.ascontiguousarray(np.transpose(tensor, transpose_tuple))
    return tensor


def generate_shape(term, sizes):
    shape = []
    for i in term:
        shape.append(sizes[i])
    return tuple(shape)

def calculate_shapes(new_term, single_term, transpose_term, sizes):
    new_shape  = generate_shape(new_term, sizes)
    single_shape = generate_shape(single_term, sizes)
    transpose_shape = generate_shape(transpose_term, sizes)
    new_size = useful_funcs.calc_new_length(new_shape)
    single_size = useful_funcs.calc_new_length(single_term)
    new_shape_sum = useful_funcs.sum_shape(new_shape)
    transpose_shape_sum = useful_funcs.sum_shape(transpose_shape)

    return {"new_shape": new_shape, 
            "single_shape": single_shape,
            "transpose_shape": transpose_shape,
            "new_size": new_size,
            "single_size": single_size,
            "new_shape_sum": new_shape_sum,
            "transpose_shape_sum": transpose_shape_sum}


def sum_entries(A, position, single_size):
    sum = 0
    for i in range(single_size):
        sum += A[position + i]
    return sum, position + single_size 


def remove_single_index(tensor, tensor_term, tensor_name, full_term_dict, sizes):
    transpose_term, new_term, single_term = build_transpose_new_term(full_term_dict["term_"+tensor_name], full_term_dict["single_"+tensor_name])
    tensor = transpose_tensor(tensor_term, transpose_term, tensor)
    tensor = tensor.reshape(-1)

    term_dict_transposed = useful_funcs.create_index_dict(transpose_term)
    shape_dict = calculate_shapes(new_term, single_term, transpose_term, sizes)

    tensor_new = np.zeros(shape_dict["new_size"])

    iterator = useful_funcs.create_iterator(shape_dict["new_shape"])

    for index in iterator:
        pos_old, pos_new = useful_funcs.calc_positions(index, shape_dict["new_shape_sum"], shape_dict["tranpose_shape_sum"], new_term, term_dict_transposed)
        tensor_new[pos_new] = sum_entries(tensor, pos_old, shape_dict["single_size"])
    
    tensor_new = np.ascontiguousarray(tensor_new.reshape(shape_dict["new_shape"]))

    return tensor_new, new_term
    

    
