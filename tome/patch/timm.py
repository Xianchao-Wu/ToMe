# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from typing import Tuple

import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer

#from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
#from tome.utils import parse_r
from merge import bipartite_soft_matching, merge_source, merge_wavg
from utils import parse_r

# /opt/conda/lib/python3.8/site-packages/timm/models/vision_transformer.py:240 for class Block
class ToMeBlock(Block): # Block = 'timm.models.vision_transformer.Block'
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications. 
        # x.shape=[1, 197, 768] -> [1, 193, 768], 
        # 原来的197=1+196,即前面有一个cls，后面是196个具体的tokens.
        #import ipdb; ipdb.set_trace()
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None # None

        x_attn, metric = self.attn(self.norm1(x), attn_size) 
        # NOTE 这里调用的是ToMeAttention, 
        # x_attn.shape=[1, 197, 768], metric.shape=k.mean(1).shape=[1, 197, 64] 
        # -> x.shape=[1, 193, 768], attn_size.shape=[1, 193, 768], attn_size.sum=197; 

        x = x + self._drop_path1(x_attn) # x.shape=[1, 197, 768]

        r = self._tome_info["r"].pop(0) # [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 12个4

        if r > 0:
            # Apply ToMe here, NOTE, 下面的逻辑最重要！
            #import ipdb; ipdb.set_trace()
            merge, unmerge = bipartite_soft_matching(
                metric, # k; [1, 197, 64] -> [1, 193, 64]
                r, # 4 -> 4
                self._tome_info["class_token"], # True
                self._tome_info["distill_token"], 
                # False, TODO what will happen if 'distill_token'=True?
            ) # NOTE 返回的merge是一个函数，需要调用的！！！ TODO

            #import ipdb; ipdb.set_trace()
            if self._tome_info["trace_source"]: # False TODO
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )

            #import ipdb; ipdb.set_trace() # NOTE
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])
            # merge=<function bipartite_soft_matching.<locals>.merge at 0x7f55073d9280>, 
            # x.shape=torch.Size([1, 197, 768]), self._tome_info['size']=None
            ##import ipdb; ipdb.set_trace()
            #y, _ = merge_wavg(unmerge, x, self._tome_info['size']) 
            # TODO debug only to learn what is 'unmerge' doing...

        x = x + self._drop_path2(self.mlp(self.norm2(x))) # now, x.shape=[1, 193, 768]
        # self.norm2 = LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        # mlp = 768 to 3072 -> gelu -> dropout -> 3072 to 768 -> dropout
        return x # [1, 193, 768]

# /opt/conda/lib/python3.8/site-packages/timm/models/vision_transformer.py:202
# for class Attention
class ToMeAttention(Attention): # Attention is 'timm.models.vision_transformer.Attention'
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention 
        # with modifications. x.shape=[1, 197, 768], size=None 
        # -> output of 'forward' = [1, 193, 768]
        #import ipdb; ipdb.set_trace()
        B, N, C = x.shape
        qkv = (
            self.qkv(x) # Linear(in_features=768, out_features=2304, bias=True), 768->768*3
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        ) # x [1, 197, 768] -> self.qkv -> [1, 197, 768*3] 
        # -> [1=Batch, 197=seq.len, 3=qkv, 12=num.heads, 64=dim] 
        # -> [3=qkv, 1=Batch, 12=num.heads, 197=seq.len, 64=dim] -> [3, 1, 12, 193, 64]
        q, k, v = (
            qkv[0], # [1, 12, 197, 64] -> [1, 12, 193, 64]
            qkv[1], # [1, 12, 197, 64] -> [1, 12, 193, 64]
            qkv[2], # [1, 12, 197, 64] -> [1, 12, 193, 64]
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale 
        # self.scale=1/sqrt(64)=1/8=0.125, 
        # attn.shape=[1, 12, 197, 197] -> [1, 12, 193, 193]

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0] 
            # [1, 1, 1, 193], NOTE 这是加在了最后一个维度上了

        attn = attn.softmax(dim=-1) # [1, 12, 197, 197]
        attn = self.attn_drop(attn) # Dropout(p=0.0, inplace=False)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) 
        # [1, 197, 768] -> [1, 12, 193, 193] * [1, 12, 193, 64] = [1, 12, 193, 64] 
        # then=transpose [1, 193, 12, 64] then=reshape [1, 193, 768]

        x = self.proj(x) # Linear(in_features=768, out_features=768, bias=True)
        x = self.proj_drop(x) # Dropout(p=0.0, inplace=False)

        # Return k as well here, 除了额外返回k.mean(1), 
        # 其他的逻辑和Attention类中的forward是一样的.
        return x, k.mean(1) # x.shape=[1, 197, 768]; 
        # k.shape=[1, 12, 197, 64] -> .mean(1) = .shape=[1, 197, 64], 
        # 这是对12个heads做了一个平均 NOTE ||| [1, 193, 64]

# e.g., transformer_class = <class 'timm.models.vision_transformer.VisionTransformer'>
def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class): # 这是构建VisionTransformer的子类
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            #import ipdb; ipdb.set_trace()
            self._tome_info["r"] = parse_r(len(self.blocks), self.r) 
            # NOTE, {'r': 0, 'size': None, 'source': None, 'trace_source': False, 
            # 'prop_attn': True, 'class_token': True, 'distill_token': False} 
            # -> append 'r':[4, ..., 4] 里面一共12个4 
            self._tome_info["size"] = None
            self._tome_info["source"] = None
            #import ipdb; ipdb.set_trace()
            return super().forward(*args, **kwdargs) # 后调用父类原来的forward函数 NOTE

    return ToMeVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), 
    set trace_source = true.

    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. 
    This is only necessary when evaluating models off

    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeVisionTransformer = make_tome_class(model.__class__) # 返回的是一个class
    # model.__class__ = <class 'timm.models.vision_transformer.VisionTransformer'>
    model.__class__ = ToMeVisionTransformer 
    # --> <class 'patch.timm.make_tome_class.<locals>.ToMeVisionTransformer'>

    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source, # False
        "prop_attn": prop_attn, # True
        "class_token": model.cls_token is not None, # model.cls_token.shape=[1, 1, 768]
        "distill_token": False,
    } # {'r': 0, 'size': None, 'source': None, 'trace_source': False, 
    # 'prop_attn': True, 'class_token': True, 'distill_token': False}

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True # not in
    # len(list(model.modules())) = 226
    for module in model.modules():
        if isinstance(module, Block): # Block is 'timm.models.vision_transformer.Block'
            module.__class__ = ToMeBlock 
            # NOTE, 'patch.timm.ToMeBlock', 这是用子类ToMeBlock来替代原来的父类Block.

            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention 
            # NOTE, 'patch.timm.ToMeAttention', 
            # 这是用子类ToMeAttention来替换掉原来的父类Attention TODO
    #import ipdb; ipdb.set_trace()
    print('changed model to ToMe!')
