import numpy as np
import f_sum_single_index
import f_single_trace
import f_map_to_bmm 
import useful_funcs

def build_sizes(term_A, term_B, shape_A, shape_B):
    sizes = {}
    for char, size in zip(term_A, shape_A):
        if char in sizes: 
            if sizes[char] != size:
                print("char", char)
                print("size", size)
                raise Exception("Unconsistend sizes.")
        else:
            sizes[char] = size
            
        
    for char, size in zip(term_B, shape_B):
        if char in sizes: 
            if sizes[char] != size:
                print("char", char)
                print("size", size)
                
                raise Exception("Unconsistend sizes.")
        else:
            sizes[char] = size
    return sizes
    
def remove_double_indices(term, Tensor, histo, sizes):
    s_term = set()
    double = {}
    for char in term:
        if char in s_term:
            if char in double:
                double[char] += 1
            else:
                double[char] = 1
        else:
            s_term.add(char)

    #update histo, invoke single_sum
    if double:
        for char in double:
            histo[char] -= double[char]
        Tensor, term = f_single_trace.single_trace(term, Tensor, sizes)

    return Tensor, term

def find_single_indices(term_A, term_B, histo):
    set_A = set()
    single_A = set()
    single_B = set()
    contract = set()
    batch = set()

    # build set of chars of term_A, remove them from histo, find singles in A
    for char in term_A:
        histo[char] -= 1
        set_A.add(char)
        if histo[char] == 0:
            single_A.add(char)
    
    #find singles in B
    for char in term_B:
        container = histo[char]
        container -= 1
        if char in set_A:
            if container == 0:
                contract.add(char)
            else:
                batch.add(char)
        else:
            if container == 0:
                single_B.add(char)
        histo[char] = container

    return single_A, single_B, contract, batch

def invoke_contraction(term_A, term_B, A, B, histo):
    sizes = build_sizes(term_A, term_B, A.shape, B.shape)
    #remove double indices
    A, term_A = remove_double_indices(term_A, A, histo, sizes)
    B, term_B = remove_double_indices(term_B, B, histo, sizes)

    #remove single indices
    single_A, single_B, contract, batch = find_single_indices(term_A, term_B, histo)
    if single_A:
        A, term_A = f_sum_single_index.remove_single_index(A, term_A, single_A, sizes)
    if single_B:
        B, term_B = f_sum_single_index.remove_single_index(B, term_B, single_B, sizes)

    #contract -> Tensor_new, term_new
    return f_map_to_bmm.map_to_bmm(term_A, term_B, A, B, contract, batch, sizes)

def build_histo(term_list):
    histo = {}
    for term in term_list:
        for char in term:
            if char in histo:
                histo[char] += 1
            else: 
                histo[char] = 1
    return histo
    
def test_ever():
    A = np.random.rand(2,3,3,4,5)
    B = np.random.rand(2,2,2,4,2,7)
    term_A = "hiijk"
    term_B = "llljho"
    term_O = "ijo"
    histo = build_histo([term_A, term_B, term_O])

    C, term_C = invoke_contraction(term_A, term_B, A, B, histo)
    print(term_C)
    
    #works!!
    inter_A = np.einsum("hiijk->hijk", A)
    inter_B = np.einsum("llljho->ljho", B)
    Test = np.einsum("hijk, ljho -> "+term_C, inter_A, inter_B)

    print("Ergebnis: ", (C-Test).sum())


test_ever()


        

#def create_double_dict(term, term_name, index_dict):
#    term_set = set()
#    double = set()
#    for i in term:
#        if i not in term_set:
#            term_set.add(i)
#        else: 
#            double.add(i)
#    index_dict["term_"+term_name] = term_set
#    index_dict["double_"+term_name] = double
#
#
#def create_single_dict(term_set, other_term_set, output_set):
#    single_set = set()
#    contract_set = set()
#    batch_set = set()
#    for i in term_set:
#        if i not in output_set:
#            if i not in other_term_set:
#                single_set.add(i)
#            else:
#                contract_set.add(i)
#        else:
#            if i in other_term_set:
#                batch_set.add(i)
#    return single_set, contract_set, batch_set
#
#
#def create_index_dict(term_A, term_B, term_O) -> dict:
#    index_dict = {}
#    index_dict["output"] = useful_funcs.create_set(term_O)
#    
#    create_double_dict(term_A, "A", index_dict)
#    create_double_dict(term_B, "B", index_dict)
#
#    index_dict["single_A"], index_dict["contract"], index_dict["batch"]= create_single_dict(index_dict["term_A"], index_dict["term_B"], index_dict["output"])
#    index_dict["single_B"], dummy_c, dummy_b = create_single_dict(index_dict["term_B"], index_dict["term_A"], index_dict["output"])
#
#    return index_dict
#
#
#def double_engine(tensor_A, tensor_B, term_A, term_B, term_O):
#    sizes = {}
#    full_term_dict = create_index_dict(term_A, term_B, term_O)
#    tensor_A, term_A, full_term_dict = f_single_engine.manage_single_tensor(tensor_A, term_A, "A", full_term_dict, sizes)
#    tensor_B, term_B, full_term_dict = f_single_engine.manage_single_tensor(tensor_B, term_B, "B", full_term_dict, sizes)
#
#    product_tensor, product_term = f_map_to_bmm.map_to_bmm(term_A, term_B, tensor_A, tensor_B, full_term_dict, sizes)
#    
#    return product_tensor, product_term

    
    

    