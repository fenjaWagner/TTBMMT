import numpy as np
import math


def transpose_tuple(term_1, term_transposed) -> tuple:
    index_dict_1 = {char: idx for idx, char in enumerate(term_1)}
    return tuple(index_dict_1[i] for i in term_transposed)


def calc_positions(index, sum_new_shape, sum_old_shape, new_term, term_dict):
    pos_A_new = sum(idx * sn for idx, sn in zip(index, sum_new_shape))
    pos_A_old = sum(
        idx * sum_old_shape[j]
        for i, idx in enumerate(index)
        for j in (term_dict[new_term[i]] if isinstance(term_dict[new_term[i]], list) else [term_dict[new_term[i]]])
    )
    return pos_A_old, pos_A_new

