import torch
from torch.nn.functional import normalize

# define a torch tensor
t = torch.tensor([1., 2., 3., -2., -5.])

# print the above tensor
print("Tensor:", t)

# normalize the tensor
t1 = normalize(t, p=1.0, dim = 0)
t2 = normalize(t, p=2.0, dim = 0)

# print normalized tensor
print("Normalized tensor with p=1:", t1)
print("Normalized tensor with p=2:", t2)