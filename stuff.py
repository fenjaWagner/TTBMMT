import numpy as np
import torch 
import mm

l = [[[1,2,3],[121,122,123],[131,132,133]],
     [[21,22,23],[221,222,223], [231,232,233]], 
     [[31,32,33],[321,322,323],[331,332,333]],
     [[41,42,43], [421,422,423], [431,432,433]]]

#A = torch.tensor(l)
#B = torch.einsum("jii-> ji", A)
#print(B)

i = [[[1,1],[1,1]],
     [[2,2],[2,2]], 
     [[3,3],[3,3]]]
j = [[8,9,10], [3,4,5]]
k = [11,11,11]



I = torch.tensor(i)
print(I.shape)

C = torch.einsum("lmi-> lm", I)
print(C)
print(C.shape)