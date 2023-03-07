import torch

# try i=j or i=j+1 merge constraints
x = torch.arange(1, 6)

to_delete = [3, 4]

mask = torch.zeros_like(x, dtype=torch.bool)

for val in to_delete:
    mask = mask | torch.eq(x, val)

new_x = torch.index_select(x, 0, (~mask).nonzero().squeeze())
print(new_x)
