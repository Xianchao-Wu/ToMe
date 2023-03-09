import torch

n = 4  # size of the first two dimensions
m = 3  # size of the third dimension
x = torch.zeros((n, n, m))  # create a n x n x m zero tensor
x[:, :, 0].fill_(float('-inf'))  # fill the first two dimensions of the first slice with -inf
x = torch.diag(torch.diag(x[:, :, 0])).unsqueeze(2).repeat(1, 1, m) + x  # set the diagonal elements to 0 in each slice

print(x)

