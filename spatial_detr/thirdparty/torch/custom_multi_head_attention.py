import torch.nn.functional as F
import math
import warnings
from typing import Optional, Tuple

import torch.nn as nn
import torch
import numpy as np


import spatial_detr.thirdparty.torch.functional as F_custom
Tensor = torch.Tensor

"""
Code is modified from torch MultiheadAttention and adapted to allow to pass a custom attention function. All credits belong to the original authors.
"""


class CustomMultiheadAttention(nn.MultiheadAttention):
    """Wrapper around torch MHA.
    This allows to use a custom attn_func
    All credits for the original code belong to the authors / contributors of pytorch.
    """
    def __init__(self, embed_dim, num_heads, dropout=0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, device, dtype)
        self.attn_gradients = None

    def save_attn_gradients(self, attn_gradients):
        print("CROSS ATT GRADS", attn_gradients.sum())
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor, query_pos: Optional[Tensor] = None, key_pos: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None, attn_func=F_custom._scaled_dot_product_attention, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
        """@see nn.MultiheadAttention for details
        """
        
        # TODO refactor currently only batch_last is supported
        assert not self.batch_first

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F_custom.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                key_pos=key_pos,
                query_pos=query_pos,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                attn_func=attn_func,
                **kwargs)
        else:
            attn_output, attn_output_weights = F_custom.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                key_pos=key_pos,
                query_pos=query_pos,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                attn_func=attn_func,
                **kwargs)

        if attn_output_weights.requires_grad:
            attn_output_weights.register_hook(self.save_attn_gradients)  # Extracts gradients during backpropagation

        return attn_output, attn_output_weights
