import opt_einsum as oe
import numpy as np
from collections import OrderedDict
import time
import numpy as np
from cgreedy import CGreedy, compute_path
import einsum_benchmark


_einsum_symbols_base = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
def get_symbol(i):
    """Get the symbol corresponding to int ``i`` - runs through the usual 52
    letters before resorting to unicode characters, starting at ``chr(192)``.

    Examples
    --------
    >>> get_symbol(2)
    'c'

    >>> get_symbol(200)
    'Ŕ'

    >>> get_symbol(20000)
    '京'
    """
    if i < 52:
        return _einsum_symbols_base[i]
    return chr(i + 140)

def rand_equation(n, reg, n_out=0, d_min=2, d_max=9, seed=None, global_dim=False, return_size_dict=False):
    """Generate a random contraction and shapes.

    Parameters
    ----------
    n : int
        Number of array arguments.
    reg : int
        'Regularity' of the contraction graph. This essentially determines how
        many indices each tensor shares with others on average.
    n_out : int, optional
        Number of output indices (i.e. the number of non-contracted indices).
        Defaults to 0, i.e., a contraction resulting in a scalar.
    d_min : int, optional
        Minimum dimension size.
    d_max : int, optional
        Maximum dimension size.
    seed: int, optional
        If not None, seed numpy's random generator with this.
    global_dim : bool, optional
        Add a global, 'broadcast', dimension to every operand.
    return_size_dict : bool, optional
        Return the mapping of indices to sizes.

    Returns
    -------
    eq : str
        The equation string.
    shapes : list[tuple[int]]
        The array shapes.
    size_dict : dict[str, int]
        The dict of index sizes, only returned if ``return_size_dict=True``.

    Examples
    --------
    >>> eq, shapes = rand_equation(n=10, reg=4, n_out=5, seed=42)
    >>> eq
    'oyeqn,tmaq,skpo,vg,hxui,n,fwxmr,hitplcj,kudlgfv,rywjsb->cebda'

    >>> shapes
    [(9, 5, 4, 5, 4),
     (4, 4, 8, 5),
     (9, 4, 6, 9),
     (6, 6),
     (6, 9, 7, 8),
     (4,),
     (9, 3, 9, 4, 9),
     (6, 8, 4, 6, 8, 6, 3),
     (4, 7, 8, 8, 6, 9, 6),
     (9, 5, 3, 3, 9, 5)]
    """

    if seed is not None:
        np.random.seed(seed)

    # total number of indices
    num_inds = n * reg // 2 + n_out
    inputs = ["" for _ in range(n)]
    output = []

    size_dict = OrderedDict((get_symbol(i), np.random.randint(d_min, d_max + 1)) for i in range(num_inds))

    # generate a list of indices to place either once or twice
    def gen():
        for i, ix in enumerate(size_dict):
            # generate an outer index
            if i < n_out:
                output.append(ix)
                yield ix
            # generate a bond
            else:
                yield ix
                yield ix

    # add the indices randomly to the inputs
    for i, ix in enumerate(np.random.permutation(list(gen()))):
        # make sure all inputs have at least one index
        if i < n:
            inputs[i] += ix
        else:
            # don't add any traces on same op
            where = np.random.randint(0, n)
            while ix in inputs[where]:
                where = np.random.randint(0, n)

            inputs[where] += ix

    # possibly add the same global dim to every arg
    if global_dim:
        gdim = get_symbol(num_inds)
        size_dict[gdim] = np.random.randint(d_min, d_max + 1)
        for i in range(n):
            inputs[i] += gdim
        output += gdim

    # randomly transpose the output indices and form equation
    output = "".join(np.random.permutation(output))
    eq = "{}->{}".format(",".join(inputs), output)

    # make the shapes
    shapes = [tuple(size_dict[ix] for ix in op) for op in inputs]

    ret = (eq, shapes)

    if return_size_dict:
        ret += (size_dict, )

    return ret

"""format_string, shapes = rand_equation(10, 3, seed=12, d_min=2, d_max=10, n_out=2)
#print(format_string)
#print(shapes)
tensors = [np.random.rand(*shape) for shape in shapes]

# initialize the CGreedy optimizer with specific parameters
optimizer = CGreedy(seed=1, minimize="size", max_repeats=1024, max_time=1.0, progbar=True, threshold_optimal=12, threads=0)

path, size_log2, flops_log10 = compute_path(format_string, *tensors, seed=1, minimize="size", max_repeats=1024,
                                            max_time=1.0, progbar=True, threshold_optimal=12, threads=0, is_linear=True)

#print(path)
ssa = einsum_benchmark.meta.runtime.to_ssa_path(path)
#print("ssa:", ssa)

annoted = einsum_benchmark.meta.runtime.to_annotated_ssa_path(format_string, ssa)
#print(annoted)

# compute a path using the optimizer with opt_einsum
#path_info = oe.contract_path(format_string, *tensors, optimize=optimizer)
#print(path_info)
"""