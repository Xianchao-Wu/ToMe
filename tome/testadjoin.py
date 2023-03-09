import torch

src_idx = torch.tensor([[[ 6],
         [42],
         [36],
         [14],
         [ 7],
         [49],
         [43],
         [13],
         [48],
         [35],
         [41],
         [21]]])

src_idx = src_idx.reshape(2, -1)

dst_idx = torch.tensor([[[ 6],
         [41],
         [35],
         [15],
         [ 6],
         [48],
         [43],
         [13],
         [48],
         [34],
         [41],
         [20]]])
dst_idx = dst_idx.reshape(2, -1)

import ipdb; ipdb.set_trace()

flag = (src_idx == dst_idx) + (src_idx == dst_idx + 1)

print(flag)
print(flag.shape)

edge_idx = torch.tensor([ 6, 42, 36, 14,  7, 49, 43, 13, 48, 35, 41, 21, 37, 28, 20, 15, 29, 34,
        50, 22, 27,  1,  8, 44, 66,  2, 31, 61, 80, 56, 60, 26, 54,  9,  3, 96,
        53, 55, 39, 89, 98, 97, 75, 85, 92, 52, 47, 19, 45, 18, 24, 23, 72, 67,
         5, 91, 95, 63, 16, 17, 82, 46, 40, 32,  4, 73, 25, 38, 10, 30, 90, 68,
        87, 78, 93, 57, 11, 74, 81, 84, 76, 12, 33, 59, 71, 94, 70, 65, 51, 83,
        69, 86, 62, 58, 77, 88, 64, 79,  0])

print(edge_idx)
print(edge_idx.shape)

import ipdb; ipdb.set_trace()

