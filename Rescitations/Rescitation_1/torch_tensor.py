import numpy
import torch

max_value = 10.0
min_value = 1.0
A = torch.rand((40,256,1000),dtype=torch.float16)
x = (max_value-min_value) * torch.rand((1000),dtype=torch.float16) + min_value

#print(test)
#print(test_smaller)


Ax = numpy.dot(A,x)

print(Ax)
print(Ax.shape) # (40,256)

