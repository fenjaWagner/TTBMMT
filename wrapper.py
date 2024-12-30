import numpy as np
import ctypes

# Load the shared library
libbmm = ctypes.CDLL('./bmm_libbmm.so')  # Use './bmm.dll' on Windows

# Define the TacoTensor structure
class TacoTensor(ctypes.Structure):
    _fields_ = [
        ('order', ctypes.c_int64),
        ('dimensions', ctypes.POINTER(ctypes.c_int64)),
        ('vals', ctypes.c_void_p),
        ('data_type', ctypes.c_int)  # DOUBLE = 0, INT = 1
    ]

# Define the compute function signature
libbmm.compute.argtypes = [
    ctypes.POINTER(TacoTensor),  # Pointer to TacoTensor for C
    ctypes.POINTER(TacoTensor),  # Pointer to TacoTensor for A
    ctypes.POINTER(TacoTensor)   # Pointer to TacoTensor for B
]
libbmm.compute.restype = ctypes.c_int

# Helper function to wrap taco_tensor_t
def call_cpp_bmm(A: np.array, B: np.array, data_t: int) -> np.array:
    # Prepare data
    shape_A = np.array(A.shape, dtype=np.int64)
    shape_B = np.array(B.shape, dtype=np.int64)
    C = np.zeros((A.shape[0], A.shape[1], B.shape[2]), dtype=np.float64)

    shape_C = np.array(C.shape, dtype=np.int64)
    
    # Flatten arrays
    A = np.ascontiguousarray(A, dtype=np.float64).flatten()
    B = np.ascontiguousarray(B, dtype=np.float64).flatten()
    C = np.ascontiguousarray(C, dtype=np.float64).flatten()

    # Wrap tensors in TacoTensor
    tensor_A = TacoTensor(
        order=3,
        dimensions=shape_A.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        vals=A.ctypes.data_as(ctypes.c_void_p),
        data_type = data_t  # DOUBLE = 0
    )

    tensor_B = TacoTensor(
        order=3,
        dimensions=shape_B.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        vals=B.ctypes.data_as(ctypes.c_void_p),
        data_type= data_t  # DOUBLE = 0
    )

    tensor_C = TacoTensor(
        order=3,
        dimensions=shape_C.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        vals=C.ctypes.data_as(ctypes.c_void_p),
        data_type= data_t # DOUBLE = 0
    )

    # Call the compute function
    result = libbmm.compute(
        ctypes.pointer(tensor_C),
        ctypes.pointer(tensor_A),
        ctypes.pointer(tensor_B)
    )
    if result != 0:
        raise RuntimeError("C++ compute function failed")

    # Reshape and return C
    return C.reshape(shape_C)

