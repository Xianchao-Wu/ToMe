# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple

import torch


def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor, # [1, 197, 64]
    r: int, # 4
    class_token: bool = False, # True
    distill_token: bool = False, # False
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    import ipdb; ipdb.set_trace()
    protected = 0
    if class_token: # True
        protected += 1
    if distill_token: # False
        protected += 1 

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1] # [1, 197, 64] -> t=197 ||| [1, 193, 64], t=193
    r = min(r, (t - protected) // 2) # r = 4

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad(): # metric = k.mean(1) = why use this TODO strange...
        metric = metric / metric.norm(dim=-1, keepdim=True) # [1, 197, 64] / [1, 197, 1] ||| [1, 193, 64] / [1, 193, 1]
        a, b = metric[..., ::2, :], metric[..., 1::2, :] # a.shape=[1, 99, 64], b.shape=[1, 98, 64] ||| a.shape=[1, 97, 64], b.shape=[1, 96, 64]
        scores = a @ b.transpose(-1, -2) # [1, 99, 64] * [1, 64, 98] -> [1, 99, 98] ||| [1, 97, 96]

        if class_token: # True
            scores[..., 0, :] = -math.inf # 每个seq中的第0个token
        if distill_token:
            scores[..., :, 0] = -math.inf # TODO

        node_max, node_idx = scores.max(dim=-1) # node_max.shape=[1, 99]; node_idx.shape=[1, 99]
        # node_max: 索引0到98，左边的第i个节点；取值=对于左边的第i个节点，右边的98个节点中，和i相似度最大的，右边第j个节点；值=i和j的相似度.
        # node_idx: 索引0到98，左边的第i个节点，右边和i最大的那个索引为j的节点，值=j
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # 降序排列, 越小的值，越需要保留！ 反过来，相似度越大的，越需要合并; edge_idx.shape=[1, 99] to [1, 99, 1]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens, [1, 95, 1], 去掉4个，保留95个。原本一共99个。
        src_idx = edge_idx[..., :r, :]  # Merged Tokens, [1, 4, 1], 这是去掉的四个tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx) # [1, 4, 1], values=[6, 41, 35, 13]; e.g., src_idx=[6, 42, 36, 14]; dst_idx=[6, 41, 35, 13]; 约束= for i, j=i or j=i-1 is okay NOTE 只合并相邻的两个token，按照这个走。对称搞一下?

        if class_token: # True
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0] # [1, 95, 1] TODO 左边的index重新从小到大排序了. 对于语音非常重要
    # x.shape=[1, 197, 768], mode='sum', 
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        import ipdb; ipdb.set_trace()
        src, dst = x[..., ::2, :], x[..., 1::2, :] # src.shape=[1, 99, 768], dst.shape=[1, 98, 768]
        n, t1, c = src.shape # 1, 99, 768
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c)) # NOTE [1, 95, 1] -> expand -> [1, 95, 768], 最后的768，对于95行，每一列都一样! NOTE 
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c)) # [1, 4, 768]
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode) # [1, 98, 768] -> out dst.shape=[1, 98, 768]

        if distill_token: # False
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1) # [1, 95, 768], [1, 98, 768] -> [1, 193, 768]

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        import ipdb; ipdb.set_trace()
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def kth_bipartite_soft_matching(
    metric: torch.Tensor, k: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (every kth element, the rest).
    If n is the number of tokens, resulting number of tokens will be n // z.

    Input size is [batch, tokens, channels].
    z indicates the stride for the first set.
    z = 2 is equivalent to regular bipartite_soft_matching with r = 0.5 * N
    """
    import ipdb; ipdb.set_trace()
    if k <= 1:
        return do_nothing, do_nothing

    def split(x):
        import ipdb; ipdb.set_trace()
        t_rnd = (x.shape[1] // k) * k
        x = x[:, :t_rnd, :].view(x.shape[0], -1, k, x.shape[2])
        a, b = (
            x[:, :, : (k - 1), :].contiguous().view(x.shape[0], -1, x.shape[-1]),
            x[:, :, (k - 1), :],
        )
        return a, b

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        r = a.shape[1]
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        import ipdb; ipdb.set_trace()
        src, dst = split(x)
        n, _, c = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        import ipdb; ipdb.set_trace()
        n, _, c = x.shape
        dst = x

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)).to(x.dtype)

        src = src.view(n, -1, (k - 1), c)
        dst = dst.view(n, -1, 1, c)

        out = torch.cat([src, dst], dim=-2)
        out = out.contiguous().view(n, -1, c)

        return out

    return merge, unmerge


def random_bipartite_soft_matching(
    metric: torch.Tensor, r: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (r chosen randomly, the rest).
    Input size is [batch, tokens, channels].

    This will reduce the number of tokens by r.
    """
    import ipdb; ipdb.set_trace()
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        B, N, _ = metric.shape
        rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)

        a_idx = rand_idx[:, :r, :]
        b_idx = rand_idx[:, r:, :]

        def split(x):
            C = x.shape[-1]
            a = x.gather(dim=1, index=a_idx.expand(B, r, C))
            b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
            return a, b

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        import ipdb; ipdb.set_trace()
        src, dst = split(x)
        C = src.shape[-1]
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        import ipdb; ipdb.set_trace()
        C = x.shape[-1]
        dst = x
        src = dst.gather(dim=-2, index=dst_idx.expand(B, r, C))

        out = torch.zeros(B, N, C, device=x.device, dtype=x.dtype)

        out.scatter_(dim=-2, index=a_idx.expand(B, r, C), src=src)
        out.scatter_(dim=-2, index=b_idx.expand(B, N - r, C), src=dst)

        return out

    return merge, unmerge

# x.shape=[1, 197, 768], size=None
def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    import ipdb; ipdb.set_trace()
    if size is None: # None
        size = torch.ones_like(x[..., 0, None]) # [1, 197, 1] all = 1

    x = merge(x * size, mode="sum") # x.shape=[1, 193, 768]
    size = merge(size, mode="sum") # size.shape=[1, 193, 1]

    x = x / size # 这是做一个平均化，size里，有些地方是2，有些是1
    return x, size # x.shape=[1, 193, 768], size.shape=[1, 193, 1]


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    import ipdb; ipdb.set_trace()
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source

