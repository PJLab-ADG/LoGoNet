from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from al3d_det.models.ops.modules import MSDeformAttn

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class DeformTransLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=4, n_points=8, 
                 light=True, norm=True, version='v1'):
        super().__init__()
        self.light = light
        self.norm = norm
        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        if self.norm:
            self.norm1 = nn.LayerNorm(d_model)

        # ffn
        if self.light == False:
            self.linear1 = nn.Linear(d_model, d_ffn)
            self.activation = _get_activation_fn(activation)
            self.dropout2 = nn.Dropout(dropout)
            self.linear2 = nn.Linear(d_ffn, d_model)
            self.dropout3 = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src_feat, reference_points, key_feat, spatial_shapes, level_start_index, padding_mask=None, query_pos=None):
        # self attention
        query_feat = self.with_pos_embed(src_feat, query_pos)
        src2 = self.self_attn(query_feat, reference_points, key_feat, spatial_shapes, level_start_index, padding_mask)
        src_feat = src_feat + self.dropout1(src2)
        if self.norm:
            src_feat = self.norm1(src_feat)

        # # ffn
        if self.light == False:
            src_feat = self.forward_ffn(src_feat)
        # src2 = self.self_attn(src_feat, reference_points, key_feat, spatial_shapes, level_start_index, padding_mask)
        # # self attention
        # src2 = self.self_attn(query_feat, reference_points, key_feat, spatial_shapes, level_start_index, padding_mask)
        # src2 = self.dropout1(src2)
        # src2 = self.norm1(src2)

        # # ffn
        # src2 = self.forward_ffn(src2)

        return src_feat