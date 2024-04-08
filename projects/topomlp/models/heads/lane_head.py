# ------------------------------------------------------------------------
# Modified from PETRv2 (https://github.com/megvii-research/PETR)
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from OpenLane-V2 (https://github.com/OpenDriveLab/OpenLane-V2)
# Copyright (c) The OpenLane-V2 Dataset Authors. All rights reserved.
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer, bias_init_with_prob
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import force_fp32
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.topomlp.core.bbox.util import normalize_bbox
import numpy as np
from mmcv.cnn import xavier_init, constant_init, kaiming_init
import math
from math import factorial
from mmdet.models.utils import NormedLinear

def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb


def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    # pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    # pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb

@HEADS.register_module()
class LaneHead(AnchorFreeHead):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_lane=300,
                 num_lanes_one2one=300,
                 k_one2many=5,
                 lambda_one2many=1.0,
                 num_reg_dim=4,
                 num_reg_fcs=2,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 positional_encoding=dict(type='SinePositionalEncoding', num_feats=128, normalize=True),
                 code_weights=None,
                 loss_cls=dict(type='CrossEntropyLoss', bg_cls_weight=0.1, use_sigmoid=False, loss_weight=1.0,
                               class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                     assigner=dict(type='HungarianAssigner', cls_cost=dict(type='ClassificationCost', weight=1.),
                                   reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                                   iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
                 test_cfg=dict(max_per_img=100),
                 with_position=True,
                 with_multiview=False,
                 depth_step=0.8,
                 depth_num=64,
                 LID=False,
                 depth_start=1,
                 position_range=[-65, -65, -8.0, 65, 65, 8.0],
                 init_cfg=None,
                 normedlinear=False,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        self.code_weights = self.code_weights[:self.code_size]
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is LaneHead):
            assert isinstance(class_weight, float), 'Expected ' \
                                                    'class_weight to have type float. Found ' \
                                                    f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                                                     'bg_cls_weight to have type float. Found ' \
                                                     f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided ' \
                                            'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_bbox['loss_weight'] == assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                           'should be exactly the same.'
            assert loss_iou['loss_weight'] == assigner['iou_cost']['weight'], \
                'The regression iou weight for loss and matcher should be' \
                'exactly the same.'
            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.num_lane = num_lane
        self.num_lanes_one2one = num_lanes_one2one
        self.k_one2many = k_one2many
        self.lambda_one2many = lambda_one2many
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.num_reg_dim = num_reg_dim
        assert self.num_reg_dim % 3 == 0
        self.bev_range = kwargs['bev_range']
        self.num_control_points = int(self.num_reg_dim / 3)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = 256
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = 3 * self.depth_num
        self.position_range = position_range
        self.LID = LID
        self.depth_start = depth_start
        self.position_level = 0
        self.with_position = with_position
        self.with_multiview = with_multiview
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
                                                 f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
                                                 f' and {num_feats}.'
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.num_pred = transformer.decoder.num_layers
        self.normedlinear = normedlinear
        super(LaneHead, self).__init__(num_classes, in_channels, init_cfg=init_cfg)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""

        self.input_proj = Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1)

        lane_cls_branch = []
        for _ in range(self.num_reg_fcs):
            lane_cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            lane_cls_branch.append(nn.LayerNorm(self.embed_dims))
            lane_cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            lane_cls_branch.append(NormedLinear(self.embed_dims, self.num_classes))
        else:
            lane_cls_branch.append(Linear(self.embed_dims, self.num_classes))
        lane_cls_branch = nn.Sequential(*lane_cls_branch)

        lane_reg_branch = []
        for _ in range(self.num_reg_fcs):
            lane_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            lane_reg_branch.append(nn.ReLU())
        lane_reg_branch.append(Linear(self.embed_dims, self.num_reg_dim))
        lane_reg_branch = nn.Sequential(*lane_reg_branch)

        self.lane_cls_branches = nn.ModuleList(
            [lane_cls_branch for _ in range(self.num_pred)])
        self.lane_reg_branches = nn.ModuleList(
            [lane_reg_branch for _ in range(self.num_pred)])

        if self.with_multiview:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims * 3 // 2, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

        if self.with_position:
            self.position_encoder = nn.Sequential(
                nn.Conv2d(self.position_dim, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

        self.reference_points_lane = nn.Embedding(self.num_lane, 3)
        self.query_embedding_lane = nn.Sequential(
            nn.Linear(self.embed_dims * 3 // 2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()
        nn.init.uniform_(self.reference_points_lane.weight.data, 0, 1)
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.lane_cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def position_embeding(self, img_feats, img_metas, masks=None):
        eps = 1e-5
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        B, N, C, H, W = img_feats.shape
        coords_h = torch.arange(H, device=img_feats.device).float() * pad_h / H
        coords_w = torch.arange(W, device=img_feats.device).float() * pad_w / W

        if self.LID:
            index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats.device).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats.device).float()
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0)  # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3]) * eps)

        img2lidars = []
        for img_meta in img_metas:
            img2lidar = []
            for i in range(len(img_meta['lidar2img'])):
                img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
            img2lidars.append(np.asarray(img2lidar))
        img2lidars = np.asarray(img2lidars)
        img2lidars = coords.new_tensor(img2lidars)  # (B, N, 4, 4)

        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (
                self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (
                self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (
                self.position_range[5] - self.position_range[2])

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)
        coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)
        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B * N, -1, H, W)
        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)

        return coords_position_embeding.view(B, N, self.embed_dims, H, W), coords_mask

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is LaneHead:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                # '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def forward(self, mlvl_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        x = mlvl_feats
        batch_size, num_cams = x.size(0), x.size(1)
        input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]
        masks = x.new_ones(
            (batch_size, num_cams, input_img_h, input_img_w))
        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
                masks[img_id, cam_id, :img_h, :img_w] = 0
        x = self.input_proj(x.flatten(0, 1))
        x = x.view(batch_size, num_cams, *x.shape[-3:])
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks, size=x.shape[-2:]).to(torch.bool)

        if self.with_position:
            coords_position_embeding, _ = self.position_embeding(mlvl_feats, img_metas, masks)
            pos_embed = coords_position_embeding
            if self.with_multiview:
                sin_embed = self.positional_encoding(masks)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
            else:
                pos_embeds = []
                for i in range(num_cams):
                    xy_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(xy_embed.unsqueeze(1))
                sin_embed = torch.cat(pos_embeds, 1)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
        else:
            if self.with_multiview:
                pos_embed = self.positional_encoding(masks)
                pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).view(x.size())
            else:
                pos_embeds = []
                for i in range(num_cams):
                    pos_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(pos_embed.unsqueeze(1))
                pos_embed = torch.cat(pos_embeds, 1)

        reference_points_lane = self.reference_points_lane.weight
        query_embeds = self.query_embedding_lane(pos2posemb3d(reference_points_lane))

        reference_points_lane = reference_points_lane.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1,
                                                                                       self.num_control_points,
                                                                                       1)  # .sigmoid()

        # attn mask for Hy
        self_attn_mask = (
            torch.zeros([self.num_lane, self.num_lane, ]).bool().to(x.device)
        )
        self_attn_mask[self.num_lanes_one2one:, 0: self.num_lanes_one2one] = True
        self_attn_mask[0: self.num_lanes_one2one, self.num_lanes_one2one:] = True

        outs_dec, _ = self.transformer(x, masks, query_embeds, pos_embed, self.lane_reg_branches,
                                       decoder_self_attn_mask=[self_attn_mask, None], )
        outs_dec = torch.nan_to_num(outs_dec)

        lane_queries = outs_dec
        outputs_lane_preds = []
        outputs_lane_clses = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points_lane.clone())
            reference = reference.view(batch_size, self.num_lane, self.num_reg_dim)
            tmp = self.lane_reg_branches[lvl](lane_queries[lvl])
            outputs_lanecls = self.lane_cls_branches[lvl](lane_queries[lvl])

            tmp = tmp.view(batch_size, self.num_lane, self.num_reg_dim)
            tmp += reference
            tmp = tmp.sigmoid()
            tmp = tmp.view(batch_size, self.num_lane, self.num_reg_dim)

            outputs_coord = tmp
            outputs_lane_preds.append(outputs_coord)
            outputs_lane_clses.append(outputs_lanecls)

        all_lane_preds = torch.stack(outputs_lane_preds)  # torch.Size([6, 1, 600, 12])
        all_lane_clses = torch.stack(outputs_lane_clses)  # torch.Size([6, 1, 600, 1])

        for p in range(self.num_reg_dim // 3):
            all_lane_preds[..., 3 * p] = all_lane_preds[..., 3 * p] * (self.bev_range[3] - self.bev_range[0]) + \
                                         self.bev_range[0]
            all_lane_preds[..., 3 * p + 1] = all_lane_preds[..., 3 * p + 1] * (self.bev_range[4] - self.bev_range[1]) + \
                                             self.bev_range[1]
            all_lane_preds[..., 3 * p + 2] = all_lane_preds[..., 3 * p + 2] * (self.bev_range[5] - self.bev_range[2]) + \
                                             self.bev_range[2]

        all_lane_cls_one2one = all_lane_clses[:, :, 0: self.num_lanes_one2one, :]
        all_lane_preds_one2one = all_lane_preds[:, :, 0: self.num_lanes_one2one, :]
        all_lane_cls_one2many = all_lane_clses[:, :, self.num_lanes_one2one:, :]
        all_lane_preds_one2many = all_lane_preds[:, :, self.num_lanes_one2one:, :]
        outs_dec_one2one = outs_dec[:, :, 0: self.num_lanes_one2one, :]
        outs_dec_one2many = outs_dec[:, :, self.num_lanes_one2one:, :]

        return all_lane_cls_one2one, all_lane_preds_one2one, outs_dec_one2one, all_lane_cls_one2many, all_lane_preds_one2many, outs_dec_one2many

    def loss(self,
             all_cls_scores_list,
             all_bbox_preds_list,
             all_cls_scores_one2many_list,
             all_bbox_preds_one2many_list,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             gt_bboxes_ignore=None):

        # NOTE defaultly only the outputs from the last feature scale is used.
        all_cls_scores = all_cls_scores_list
        all_bbox_preds = all_bbox_preds_list
        assert gt_bboxes_ignore is None, \
            'Only supports for gt_bboxes_ignore setting to None.'

        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        # for one2many
        one2many_gt_bboxes_list = []
        one2many_gt_labels_list = []

        for gt_bboxes in gt_bboxes_list:
            one2many_gt_bboxes_list.append(gt_bboxes.repeat(self.k_one2many, 1))
        for gt_labels in gt_labels_list:
            one2many_gt_labels_list.append(gt_labels.repeat(self.k_one2many))
        all_gt_bboxes_list_one2many = [one2many_gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list_one2many = [one2many_gt_labels_list for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou, assign_result = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)

        img_metas_list_one2many = img_metas_list
        all_gt_bboxes_ignore_list_one2many = all_gt_bboxes_ignore_list
        losses_cls_one2many, losses_bbox_one2many, losses_iou_one2many, assign_result_one2many = multi_apply(
            self.loss_single, all_cls_scores_one2many_list, all_bbox_preds_one2many_list,
            all_gt_bboxes_list_one2many, all_gt_labels_list_one2many, img_metas_list_one2many,
            all_gt_bboxes_ignore_list_one2many)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_cls_H'] = losses_cls_one2many[-1] * self.lambda_one2many
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_bbox_H'] = losses_bbox_one2many[-1] * self.lambda_one2many
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_cls_H'] = losses_cls_one2many[num_dec_layer] * self.lambda_one2many
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_bbox_H'] = losses_bbox_one2many[num_dec_layer] * self.lambda_one2many
            num_dec_layer += 1
        return loss_dict, assign_result

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           img_metas, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, assign_result) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        bbox_preds = bbox_preds.reshape(-1, self.num_reg_dim)
        loss_iou = None

        bbox_preds = self.control_points_to_lane_points(bbox_preds)
        # bbox_targets = self.control_points_to_lane_points(bbox_targets)
        bbox_targets = self.control_points_to_lane_points(bbox_targets)
        bbox_weights = bbox_weights.mean(-1).unsqueeze(-1).repeat(1, bbox_preds.shape[-1])
        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou, assign_result

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):

        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list, pos_assigned_gt_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        assign_result = dict(
            pos_inds=pos_inds_list, neg_inds=neg_inds_list, pos_assigned_gt_inds=pos_assigned_gt_inds_list
        )
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg, assign_result)

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_bboxes,
                           gt_labels,
                           img_meta,
                           gt_bboxes_ignore=None):

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        bbox_pred = self.control_points_to_lane_points(bbox_pred)
        gt_bboxes = self.control_points_to_lane_points(gt_bboxes)
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, img_meta,
                                             gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        pos_gt_bboxes = sampling_result.pos_gt_bboxes
        pos_gt_bboxes_targets = pos_gt_bboxes

        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, pos_assigned_gt_inds)

    def get_bboxes(self,
                   all_cls_scores_list,
                   all_bbox_preds_list,
                   img_metas,
                   rescale=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        # NOTE defaultly only using outputs from the last feature level,
        # and only the outputs from the last decoder layer is used.
        cls_scores = all_cls_scores_list[-1]
        bbox_preds = all_bbox_preds_list[-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score, bbox_pred,
                                                img_shape, scale_factor,
                                                rescale)
            result_list.append(proposals)

        return result_list

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):

        assert len(cls_score) == len(bbox_pred)
        cls_score = cls_score.sigmoid()
        det_bboxes = bbox_pred
        for p in range(self.num_reg_dim // 3):
            det_bboxes[..., 3 * p].clamp_(min=self.bev_range[0], max=self.bev_range[3])
            det_bboxes[..., 3 * p + 1].clamp_(min=self.bev_range[1], max=self.bev_range[4])
            det_bboxes[..., 3 * p + 2].clamp_(min=self.bev_range[2], max=self.bev_range[5])

        return det_bboxes.cpu().numpy(), cls_score.cpu().numpy()

    def onnx_export(self, **kwargs):
        raise NotImplementedError(f'TODO: replace 4 with self.num_reg_dim : {self.num_reg_dim}')

    def control_points_to_lane_points(self, lanes):
        lanes = lanes.reshape(-1, lanes.shape[-1] // 3, 3)

        def comb(n, k):
            return factorial(n) // (factorial(k) * factorial(n - k))

        n_points = 11
        n_control = lanes.shape[1]
        A = np.zeros((n_points, n_control))
        t = np.arange(n_points) / (n_points - 1)
        for i in range(n_points):
            for j in range(n_control):
                A[i, j] = comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
        bezier_A = torch.tensor(A, dtype=torch.float32).to(lanes.device)
        lanes = torch.einsum('ij,njk->nik', bezier_A, lanes)
        lanes = lanes.reshape(lanes.shape[0], -1)

        return lanes



