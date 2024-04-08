import math
import warnings
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.cnn.bricks.drop import build_dropout
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, xavier_init)
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import (ATTENTION,TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)
import copy
import torch.utils.checkpoint as cp
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models.utils.transformer import DetrTransformerDecoderLayer
from torch import Tensor
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

# from .cosformer import CosformerAttention
from .attention import FlashMHA

@TRANSFORMER.register_module()
class PETRTransformer(BaseModule):
    """Implements the DETR transformer.
    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:
        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None, cross=False):
        super(PETRTransformer, self).__init__(init_cfg=init_cfg)
        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        else:
            self.encoder = None
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.cross = cross
        # self.target = nn.Embedding(900, self.embed_dims)

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, x, mask, query_embed, pos_embed, reg_branch=None, decoder_self_attn_mask=None, target=None):
        """Forward function for `Transformer`.
        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        bs, n, c, h, w = x.shape
       
        if self.cross:
            x = x.permute(1, 3, 4, 0, 2).reshape(-1, bs, c)  # [bs, c, h, w] -> [n*h*w, bs, c]
            pos_embed = pos_embed.permute(1, 3, 4, 0, 2).reshape(-1, bs, c)
            query_embed = query_embed.unsqueeze(1).repeat(
                1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
            mask = mask.view(bs, -1)  # [bs, n, h, w] -> [bs, n*h*w]
            if self.encoder is not None:
                memory = self.encoder(
                    query=x,
                    key=None,
                    value=None,
                    query_pos=pos_embed,
                    query_key_padding_mask=mask)
            else:
                memory = x.clone()
        else:
            x = x.view(bs*n, c, -1).permute(2, 0, 1)  # [bs, n, c, h, w] -> [h*w, bs*n, c]
            pos_embed = pos_embed.view(bs*n, c, -1).permute(2, 0, 1)
            mask = mask.view(bs*n, -1)  # [bs, n, h, w] -> [bs*n, h*w]
            if self.encoder is not None:
                memory = self.encoder(
                    query=x,
                    key=None,
                    value=None,
                    query_pos=pos_embed,
                    query_key_padding_mask=mask)
            else:
                memory = x.clone()
            
            memory = memory.reshape(h*w, bs, n, c).permute(2, 0, 1, 3).reshape(-1, bs, c) # [h*w, bs*n, c] -> [n*h*w, bs, c]
            pos_embed = pos_embed.reshape(h*w, bs, n, c).permute(2, 0, 1, 3).reshape(-1, bs, c)
            query_embed = query_embed.unsqueeze(1).repeat(
                1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
            mask = mask.reshape(bs, n, -1).view(bs, -1)  # [bs*n, h*w] -> [bs, n*h*w]

        if target is None:
            target = torch.zeros_like(query_embed)
            
        # target = torch.mean(memory, dim=0, keepdim=True).repeat(query_embed.size(0), 1, 1)
        # target = self.target.weight.unsqueeze(1).repeat(1, bs, 1)

        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
            reg_branch=reg_branch,
            attn_masks=decoder_self_attn_mask,
            )
        out_dec = out_dec.transpose(1, 2)
        memory = memory.reshape(n, h, w, bs, c).permute(3, 0, 4, 1, 2)
        return  out_dec, memory

# @TRANSFORMER.register_module()
# class PETRTransformer(BaseModule):
#     """Implements the DETR transformer.
#     Following the official DETR implementation, this module copy-paste
#     from torch.nn.Transformer with modifications:
#         * positional encodings are passed in MultiheadAttention
#         * extra LN at the end of encoder is removed
#         * decoder returns a stack of activations from all decoding layers
#     See `paper: End-to-End Object Detection with Transformers
#     <https://arxiv.org/pdf/2005.12872>`_ for details.
#     Args:
#         encoder (`mmcv.ConfigDict` | Dict): Config of
#             TransformerEncoder. Defaults to None.
#         decoder ((`mmcv.ConfigDict` | Dict)): Config of
#             TransformerDecoder. Defaults to None
#         init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
#             Defaults to None.
#     """

#     def __init__(self, encoder=None, decoder=None, init_cfg=None, cross=False):
#         super(PETRTransformer, self).__init__(init_cfg=init_cfg)
#         if encoder is not None:
#             self.encoder = build_transformer_layer_sequence(encoder)
#         else:
#             self.encoder = None
#         self.decoder = build_transformer_layer_sequence(decoder)
#         self.embed_dims = self.decoder.embed_dims
#         self.cross = cross
#         # self.target = nn.Embedding(900, self.embed_dims)

#     def init_weights(self):
#         # follow the official DETR to init parameters
#         for m in self.modules():
#             if hasattr(m, 'weight') and m.weight.dim() > 1:
#                 xavier_init(m, distribution='uniform')
#         self._is_init = True
        
#     def forward(self, x, mask, query_embed, pos_embed, reg_branch=None):
#         """Forward function for `Transformer`.
#         Args:
#             x (Tensor): Input query with shape [bs, c, h, w] where
#                 c = embed_dims.
#             mask (Tensor): The key_padding_mask used for encoder and decoder,
#                 with shape [bs, h, w].
#             query_embed (Tensor): The query embedding for decoder, with shape
#                 [num_query, c].
#             pos_embed (Tensor): The positional encoding for encoder and
#                 decoder, with the same shape as `x`.
#         Returns:
#             tuple[Tensor]: results of decoder containing the following tensor.
#                 - out_dec: Output from decoder. If return_intermediate_dec \
#                       is True output has shape [num_dec_layers, bs,
#                       num_query, embed_dims], else has shape [1, bs, \
#                       num_query, embed_dims].
#                 - memory: Output results from encoder, with shape \
#                       [bs, embed_dims, h, w].
#         """
#         bs, n, c, h, w = x.shape
#         memory = x.permute(1, 3, 4, 0, 2).reshape(-1, bs, c) # [h*w, bs*n, c] -> [n*h*w, bs, c]
#         pos_embed = pos_embed.permute(1, 3, 4, 0, 2).reshape(-1, bs, c)
#         query_embed = query_embed.unsqueeze(1).repeat(
#             1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
#         mask = mask.view(bs, -1)  # [bs*n, h*w] -> [bs, n*h*w]

#         target = torch.zeros_like(query_embed)
#         # target = torch.mean(memory, dim=0, keepdim=True).repeat(query_embed.size(0), 1, 1)
#         # target = self.target.weight.unsqueeze(1).repeat(1, bs, 1)

#         # out_dec: [num_layers, num_query, bs, dim]
#         out_dec = self.decoder(
#             query=target,
#             key=memory,
#             value=memory,
#             key_pos=pos_embed,
#             query_pos=query_embed,
#             key_padding_mask=mask,
#             reg_branch=reg_branch,
#             )
#         out_dec = out_dec.transpose(1, 2)
#         memory = memory.reshape(n, h, w, bs, c).permute(3, 0, 4, 1, 2)
#         return  out_dec, memory


@TRANSFORMER_LAYER.register_module()
class PETRTransformerDecoderLayer(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 with_cp=False,
                 **kwargs):
        super(PETRTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
        self.use_checkpoint = with_cp
    
    def _forward(self, 
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        x = super(PETRTransformerDecoderLayer, self).forward(
                query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                )

        return x

    def forward(self, 
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs
                ):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        
        # query, key, value, query_pos, key_pos, attn_masks, query_key_padding_mask, key_padding_mask

        if self.use_checkpoint and self.training:
            x = cp.checkpoint(
                self._forward, 
                query,
                key,
                value,
                query_pos,
                key_pos,
                attn_masks,
                query_key_padding_mask,
                key_padding_mask,
                )
        else:
            x = self._forward(
            query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_masks=attn_masks,
            query_key_padding_mask=query_key_padding_mask,
            key_padding_mask=key_padding_mask
            )
        
        

        return x
    




@TRANSFORMER.register_module()
class PETRTransformerCP(BaseModule):
    """Implements the DETR transformer.
    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:
        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None, cross=False):
        super(PETRTransformerCP, self).__init__(init_cfg=init_cfg)
        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        else:
            self.encoder = None
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.cross = cross
        # self.target = nn.Embedding(900, self.embed_dims)
        self.with_cp = True

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, x, mask, query_embed, pos_embed, reg_branch=None):
        """Forward function for `Transformer`.
        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        bs, n, c, h, w = x.shape
       
        if self.cross:
            x = x.permute(1, 3, 4, 0, 2).reshape(-1, bs, c)  # [bs, c, h, w] -> [n*h*w, bs, c]
            pos_embed = pos_embed.permute(1, 3, 4, 0, 2).reshape(-1, bs, c)
            query_embed = query_embed.unsqueeze(1).repeat(
                1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
            mask = mask.view(bs, -1)  # [bs, n, h, w] -> [bs, n*h*w]
            if self.encoder is not None:
                memory = self.encoder(
                    query=x,
                    key=None,
                    value=None,
                    query_pos=pos_embed,
                    query_key_padding_mask=mask)
            else:
                memory = x.clone()
        else:
            x = x.view(bs*n, c, -1).permute(2, 0, 1)  # [bs, n, c, h, w] -> [h*w, bs*n, c]
            pos_embed = pos_embed.view(bs*n, c, -1).permute(2, 0, 1)
            mask = mask.view(bs*n, -1)  # [bs, n, h, w] -> [bs*n, h*w]
            if self.encoder is not None:
                memory = self.encoder(
                    query=x,
                    key=None,
                    value=None,
                    query_pos=pos_embed,
                    query_key_padding_mask=mask)
            else:
                memory = x.clone()
            
            memory = memory.reshape(h*w, bs, n, c).permute(2, 0, 1, 3).reshape(-1, bs, c) # [h*w, bs*n, c] -> [n*h*w, bs, c]
            pos_embed = pos_embed.reshape(h*w, bs, n, c).permute(2, 0, 1, 3).reshape(-1, bs, c)
            query_embed = query_embed.unsqueeze(1).repeat(
                1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
            mask = mask.reshape(bs, n, -1).view(bs, -1)  # [bs*n, h*w] -> [bs, n*h*w]

        target = torch.zeros_like(query_embed)
        # target = torch.mean(memory, dim=0, keepdim=True).repeat(query_embed.size(0), 1, 1)
        # target = self.target.weight.unsqueeze(1).repeat(1, bs, 1)

        # out_dec: [num_layers, num_query, bs, dim]

        # out_dec = self.decoder(
        #     query=target,
        #     key=memory,
        #     value=memory,
        #     key_pos=pos_embed,
        #     query_pos=query_embed,
        #     key_padding_mask=mask,
        #     reg_branch=reg_branch,
        #     )

        def _inner_forward(target):
            out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
            reg_branch=reg_branch,
            )
            return out_dec

        if self.with_cp:
            out_dec = cp.checkpoint(_inner_forward, target)
        else:
            out_dec = _inner_forward(target)

        out_dec = out_dec.transpose(1, 2)
        memory = memory.reshape(n, h, w, bs, c).permute(3, 0, 4, 1, 2)
        return  out_dec, memory

class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        k_dim: total number of features in key. Default: None.
        v_dim: total number of features in value. Default: None.
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
    Note that if :attr:`k_dim` and :attr:`v_dim` are None, they will be set
    to :attr:`embed_dim` such that query, key, and value have the same
    number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim: int, num_heads: int,
                 attention_dropout: Optional[float] = 0.,
                 scale_factor: Optional[float] = 1.,
                 bias: Optional[bool] = True,
                 add_bias_kv: Optional[bool] = False,
                 add_zero_attn: Optional[bool] = False,
                 k_dim: Optional[int] = None, v_dim: Optional[int] = None,
                 batch_first: Optional[bool] = False,
                 **kwargs: Dict[str, Any]) -> None:
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.k_dim = k_dim if k_dim is not None else self.embed_dim
        self.v_dim = v_dim if v_dim is not None else self.embed_dim
        self._qkv_same_embed_dim = self.embed_dim == self.k_dim == self.v_dim

        self.num_heads = num_heads
        self.scale_factor = scale_factor
        self.batch_first = batch_first
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = float(self.head_dim * self.scale_factor) ** -0.5
        if not self.head_dim * self.num_heads == self.embed_dim:
            raise ValueError(f"embed_dim {self.embed_dim} not divisible by num_heads {self.num_heads}")

        self.in_proj = nn.Linear(self.embed_dim, self.embed_dim + self.k_dim + self.v_dim, bias=bias)
        self.dropout = nn.Dropout(attention_dropout)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, self.embed_dim)))
            self.bias_v = nn.Parameter(torch.empty((1, 1, self.embed_dim)))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj.weight)

        if self.in_proj.bias is not None:
            nn.init.constant_(self.in_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                attn_bias: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                need_weights: Optional[bool] = True,
                static_k: Optional[Tensor] = None,
                static_v: Optional[Tensor] = None,) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        attn_bias: 2D or 3D mask that add bias to attention output weights. Used for relative positional embedding.
            A 2D bias will be broadcasted for all the batches while a 3D mask allows to specify a different mask for
            the entries of each batch.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        static_k, static_v: static key and value used for attention operators.
    Shapes for inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
        - attn_bias: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
            source sequence length.
            If a 3D mask: :math:`(N\cdot\text{num\_heads}, L, S)` where N is the batch size, L is the target sequence
            length, S is the source sequence length. ``attn_bias`` allows to pass pos embed directly into attention
            If a ByteTensor is provided, the non-zero positions are not allowed to attend while the zero positions will
            be unchanged. If a BoolTensor is provided, positions with ``True`` is not allowed to attend while ``False`` 
            values will be unchanged. If a FloatTensor is provided, it will be added to the attention weight.
        - attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
            source sequence length.
            If a 3D mask: :math:`(N\cdot\text{num\_heads}, L, S)` where N is the batch size, L is the target sequence
            length, S is the source sequence length. ``attn_mask`` ensure that position i is allowed to attend
            the unmasked positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the position
            with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
            N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
            N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
    Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
            L is the target sequence length, S is the source sequence length.
        """
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        if not key.shape[:2] == value.shape[:2]:
            raise ValueError(f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}")

        q, k, v = self.in_projection(query, key, value)

        # prep attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                attn_mask = attn_mask.to(torch.bool)
            else:
                assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                    f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        # prep key padding mask
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)

        # add bias along batch dimension (currently second)
        if self.bias_k is not None and self.bias_v is not None:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert self.bias_k is None
            assert self.bias_v is None

        #
        # reshape q, k, v for multihead attention and make em batch first
        #
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if static_k is None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert static_k.size(0) == bsz * self.num_heads, \
                f"expecting static_k.size(0) of {bsz * self.num_heads}, but got {static_k.size(0)}"
            assert static_k.size(2) == self.head_dim, \
                f"expecting static_k.size(2) of {self.head_dim}, but got {static_k.size(2)}"
            k = static_k
        if static_v is None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert static_v.size(0) == bsz * self.num_heads, \
                f"expecting static_v.size(0) of {bsz * self.num_heads}, but got {static_v.size(0)}"
            assert static_v.size(2) == self.head_dim, \
                f"expecting static_v.size(2) of {self.head_dim}, but got {static_v.size(2)}"
            v = static_v

        # add zero attention along batch dimension (now first)
        if self.add_zero_attn:
            zero_attn_shape = (bsz * self.num_heads, 1, self.head_dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        # update source sequence length after adjustments
        src_len = k.size(1)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
                expand(-1, self.num_heads, -1, -1).reshape(bsz * self.num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        #
        # (deep breath) calculate attention and out projection
        #
        attn_output, attn_output_weights = self.attention(q, k, v, attn_bias, attn_mask)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_projection(attn_output)

        attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len) if need_weights else None

        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def in_projection(self, q: Tensor, k: Tensor, v: Tensor) -> List[Tensor]:
        r"""
        Performs the in-projection step of the attention operation, using packed weights.
        Output is a triple containing projection tensors for query, key and value.
        Args:
            q, k, v: query, key and value tensors to be projected. For self-attention,
                these are typically the same tensor; for encoder-decoder attention,
                k and v are typically the same tensor. (We take advantage of these
                identities for performance if they are present.) Regardless, q, k and v
                must share a common embedding dimension; otherwise their shapes may vary.
        Shape:
            Inputs:
            - q: :math:`(..., E)` where E is the embedding dimension
            - k: :math:`(..., E)` where E is the embedding dimension
            - v: :math:`(..., E)` where E is the embedding dimension
            Output:
            - in output list :math:`[q', k', v']`, each output tensor will have the
                same shape as the corresponding input tensor.
        """
        if k is v:
            # self-attention
            if q is k:
                return self.in_proj(q).split((self.embed_dim, self.k_dim, self.v_dim), dim=-1)
            # encoder-decoder attention
            else:
                w_q, w_kv = self.in_proj.weight.split([self.embed_dim, self.k_dim + self.v_dim])
                b_q, b_kv = None if self.in_proj.bias is None else self.in_proj.bias.split([self.embed_dim, self.k_dim + self.v_dim])
                return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).split((self.k_dim, self.v_dim), dim=-1)
        else:
            w_q, w_k, w_v = self.in_proj.weight.split([self.embed_dim, self.k_dim, self.v_dim])
            b_q, b_k, b_v = None if self.in_proj.bias is None else self.in_proj.bias.split([self.embed_dim, self.k_dim, self.v_dim])
            return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

    def attention(self, q: Tensor, k: Tensor, v: Tensor, attn_bias: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None,) -> Tuple[Tensor, Tensor]:
        r"""
        Computes scaled dot product attention on query, key and value tensors, using
        an optional attention mask if passed, and applying dropout if a probability
        greater than 0.0 is specified.
        Returns a tensor pair containing attended values and attention weights.
        Args:
            q, k, v: query, key and value tensors. See Shape section for shape details.
            attn_mask: optional tensor containing mask values to be added to calculated
                attention. May be 2D or 3D; see Shape section for details.
            attn_bias: optional tensor containing bias values to be added to calculated
                attention. Used for relative positional embedding. May be 2D or 3D; see
                Shape section for details.
        Shape:
            - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
                and E is embedding dimension.
            - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                and E is embedding dimension.
            - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                and E is embedding dimension.
            - attn_bias: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
                shape :math:`(Nt, Ns)`.
            - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
                shape :math:`(Nt, Ns)`.
            - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
                have shape :math:`(B, Nt, Ns)`
        """
        q = q * self.scaling
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn = torch.bmm(q, k.transpose(-2, -1))
        if attn_bias is not None:
            attn += attn_bias
        if attn_mask is not None:
            attn += attn_mask
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(attn, v)
        return output, attn

        # q *= self.scaling
        # tokens = q.size(1)
        # output = []
        # attention = []
        # seq_len = 10
        
        # for token_index_query in range(0, tokens, seq_len):        
        #     attn = torch.bmm(q[:, token_index_query:token_index_query+seq_len], k.transpose(-2, -1))
        #     if attn_bias is not None:
        #         attn += attn_bias
        #     if attn_mask is not None:
        #         attn += attn_mask
        #     # attn = F.softmax(attn, dim = -1)
        #     attn = attn.softmax(dim = -1)
        #     attn = self.dropout(attn)
        #     # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        #     out_index = torch.bmm(attn, v)
        #     output.append(out_index)
        #     attention.append(attn)
        # output = torch.cat(output, dim=1)
        # attn = torch.cat(attention, dim=1)
        # return output, attn

    def out_projection(self, attn_output: Tensor) -> Tensor:
        return self.out_proj(attn_output)

@ATTENTION.register_module()
class PETRMultiheadAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.
    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        super(PETRMultiheadAttention, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)
        # self.attn = MultiheadAttention(embed_dims, num_heads, attn_drop,
        #                                   **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.
        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.
        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
            # value = value+ key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))


@ATTENTION.register_module()
class PETRMultiheadFlashAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.
    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=True,
                 **kwargs):
        super(PETRMultiheadFlashAttention, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = True

        self.attn = FlashMHA(embed_dims, num_heads, attn_drop, dtype=torch.float16, device='cuda',
                             **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.
        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.
        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(
            q=query,
            k=key,
            v=value,
            key_padding_mask=None)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))


@ATTENTION.register_module()
class DecouplePETRMultiheadAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.
    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 num_cams=6,
                 batch_first=False,
                 **kwargs):
        super(DecouplePETRMultiheadAttention, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.num_cams = num_cams

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)
        # self.attention_weights = nn.Linear(embed_dims, num_cams)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.
        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.
        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
            # value = value+ key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        n, bs, c = key.size()
        key = key.view(self.num_cams, -1, bs, c)
        value = value.view(self.num_cams, -1, bs, c)
        key_padding_mask = key_padding_mask.view(bs, self.num_cams, -1)
        outviews = []
        for i in range(self.num_cams):
            out_view = self.attn(
                query=query,
                key=key[i],
                value=value[i],
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask[:,i,:])[0]
            outviews.append(out_view)
        out = sum(outviews)

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))

@ATTENTION.register_module()
class CosMultiheadAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.
    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        super(CosMultiheadAttention, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = CosformerAttention(embed_dims, num_heads, dropout_rate = attn_drop,
                                          **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.
        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.
        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
            # value = value+ key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask
            )

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class PETRTransformerEncoder(TransformerLayerSequence):
    """TransformerEncoder of DETR.
    Args:
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`. Only used when `self.pre_norm` is `True`
    """

    def __init__(self, *args, post_norm_cfg=dict(type='LN'), **kwargs):
        super(PETRTransformerEncoder, self).__init__(*args, **kwargs)
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(
                post_norm_cfg, self.embed_dims)[1] if self.pre_norm else None
        else:
            assert not self.pre_norm, f'Use prenorm in ' \
                                      f'{self.__class__.__name__},' \
                                      f'Please specify post_norm_cfg'
            self.post_norm = None

    def forward(self, *args, **kwargs):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        x = super(PETRTransformerEncoder, self).forward(*args, **kwargs)
        if self.post_norm is not None:
            x = self.post_norm(x)
        return x


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class PETRTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 post_norm_cfg=dict(type='LN'),
                 return_intermediate=False,
                 **kwargs):

        super(PETRTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None

    def forward(self, query, *args, **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        if not self.return_intermediate:
            x = super().forward(query, *args, **kwargs)
            if self.post_norm:
                x = self.post_norm(x)[None]
            return x

        intermediate = []
        # print("input", query.size())
        for layer in self.layers:
            query = layer(query, *args, **kwargs)
            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)
            # print(query.size())
        return torch.stack(intermediate)

# SQR
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class PETRTransformerDecoder_SQR(PETRTransformerDecoder):
    """Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 return_intermediate=False,
                 start_q=None,
                 end_q=None,
                 **kwargs):

        super(PETRTransformerDecoder_SQR, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.start_q = start_q
        self.end_q = end_q

    def forward(self, query, *args, **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        # inference forward function is unchanged for SQR
        if not kwargs['query_pos'].requires_grad:
            return super(PETRTransformerDecoder_SQR, self).forward(query, *args, **kwargs)

        # Training forward starts here
        if not self.return_intermediate:
            x = super().forward(query, *args, **kwargs)
            if self.post_norm:
                x = self.post_norm(x)[None]
            return x

        key, value, query_pos, key_padding_mask, key_pos = kwargs['key'], \
                                             kwargs['value'], \
                                             kwargs['query_pos'], \
                                             kwargs['key_padding_mask'], \
                                             kwargs['key_pos']

        intermediate = []
        query_list_reserve = [query]
        batchsize = query.shape[1]
        # import ipdb;ipdb.set_trace()
        for lid, layer in enumerate(self.layers):

            start_q = self.start_q[lid]
            end_q = self.end_q[lid]
            query_list = query_list_reserve.copy()[start_q:end_q]

            # prepare for parallel process
            output = torch.cat(query_list, dim=1)
            fakesetsize = int(output.shape[1] / batchsize)

            # import ipdb;ipdb.set_trace()  
            # dict_keys(['key', 'value', 'key_pos', 'query_pos', 'key_padding_mask', 'reg_branch', 'attn_masks']
            kwargs['key_pos'] = key_pos.repeat(1, fakesetsize, 1)
            kwargs['key'] = key.repeat(1, fakesetsize, 1)
            kwargs['value'] = value.repeat(1, fakesetsize, 1)
            kwargs['query_pos'] = query_pos.repeat(1, fakesetsize, 1)
            kwargs['key_padding_mask'] = key_padding_mask.repeat(fakesetsize, 1)

            output = layer(output, *args, **kwargs)

            for i in range(fakesetsize):
                query_list_reserve.append(output[:, batchsize*i:batchsize*(i+1), :])

            if self.return_intermediate:
                if self.post_norm is not None:
                    for i in range(fakesetsize):
                        intermediate.append(self.post_norm(output[:, batchsize*i:batchsize*(i+1), :]))
                else:
                    for i in range(fakesetsize):
                        intermediate.append(output[:, batchsize*i:batchsize*(i+1), :])
            # print(query.size())
        return torch.stack(intermediate)



@TRANSFORMER_LAYER_SEQUENCE.register_module()
class PETRTransformerDecoderCP(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 post_norm_cfg=dict(type='LN'),
                 return_intermediate=False,
                 with_cp = True,
                 **kwargs):

        super(PETRTransformerDecoderCP, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None
        self.with_cp = with_cp

    def forward(self, query, *args, **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        if not self.return_intermediate:
            x = super().forward(query, *args, **kwargs)
            if self.post_norm:
                x = self.post_norm(x)[None]
            return x
        
        def _inner_forward(layer, query):
            out = layer(query, *args, **kwargs)
            return out

        intermediate = []
        # print("input", query.size())
        for layer in self.layers:

            def _inner_forward(query):
                out = layer(query, *args, **kwargs)
                return out

            if self.with_cp and self.training:
                query = cp.checkpoint(_inner_forward, query)
            else:
                query = _inner_forward(query)

            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)
            # print(query.size())
        return torch.stack(intermediate)




# query = torch.rand(100, 1, 256)
# key = torch.rand(1200, 1, 256)
# value= torch.rand(1200, 1, 256)
# model = DecouplePETRMultiheadAttention(256, 8)
# outputs = model(query, key, value)
# print(outputs.size())
