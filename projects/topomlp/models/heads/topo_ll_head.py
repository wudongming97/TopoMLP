import copy

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from mmcv.cnn import Linear, bias_init_with_prob, build_activation_layer
from mmcv.cnn.bricks.transformer import build_feedforward_network
from mmcv.runner import auto_fp16, force_fp32
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid
from math import factorial

from mmcv.cnn.bricks.registry import (ATTENTION, FEEDFORWARD_NETWORK, POSITIONAL_ENCODING,
                                      TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE)

from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)


def build_attention(cfg, default_args=None):
    """Builder for attention."""
    return build_from_cfg(cfg, ATTENTION, default_args)


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@HEADS.register_module()
class TopoLLHead(nn.Module):
    def __init__(self,
                 in_channels_o1,
                 in_channels_o2=None,
                 shared_param=False,
                 loss_rel=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25),
                 loss_ll_l1_weight=0,
                 add_lane_pred=False,
                 lane_pred_dimension=12,
                 is_detach=False):
        super().__init__()

        self.MLP_o1 = MLP(in_channels_o1, in_channels_o1, 128, 3)
        self.shared_param = shared_param
        if shared_param:
            self.MLP_o2 = self.MLP_o1
        else:
            self.MLP_o2 = MLP(in_channels_o2, in_channels_o2, 128, 3)

        self.classifier = MLP(256, 256, 1, 3)

        self.loss_rel = build_loss(loss_rel)

        self.add_lane_pred = add_lane_pred
        self.lane_pred_dimension = lane_pred_dimension
        if self.add_lane_pred:
            self.lane_mlp1 = MLP(self.lane_pred_dimension, 128, 128, 1)
            self.lane_mlp2 = MLP(self.lane_pred_dimension, 128, 128, 1)

        self.is_detach = is_detach
        self.loss_ll_l1_weight = loss_ll_l1_weight

    def forward_train(self, o1_feats, o1_assign_results, o2_feats, o2_assign_results, gt_adj):
        rel_pred = self.forward(o1_feats, o2_feats)
        losses = self.loss(rel_pred, gt_adj, o1_assign_results, o2_assign_results)
        return losses

    def get_topology(self, pred_adj_list):
        pred_adj = pred_adj_list.squeeze(-1).sigmoid()
        # pred_adj = pred_adj + 0.5

        # pred_adj_index = pred_adj > 0.5
        # pred_adj[pred_adj_index] = 1.0
        # pred_adj_index_neg = pred_adj <= 0.5
        # pred_adj[pred_adj_index_neg] = 0.0
        return pred_adj.cpu().numpy()

    def forward(self, o1_feats, o2_feats, o1_pos, o2_pos, img_metas=None):
        # feats: [D, B, num_query, num_embedding]
        o1_feat = o1_feats[-1].clone()
        o2_feat = o2_feats[-1].clone()
        o1_pos = o1_pos[-1].clone()
        o2_pos = o2_pos[-1].clone()

        if self.is_detach:
            o1_pos = o1_pos.detach()
            o2_pos = o2_pos.detach()
            o1_feat = o1_feat.detach()
            o2_feat = o2_feat.detach()

        o1_embeds = self.MLP_o1(o1_feat)
        o2_embeds = self.MLP_o2(o2_feat)
        if self.add_lane_pred:
            o1_embeds = o1_embeds + self.lane_mlp1(o1_pos)
            o2_embeds = o2_embeds + self.lane_mlp2(o2_pos)

        num_query_o1 = o1_embeds.size(1)
        num_query_o2 = o2_embeds.size(1)
        o1_tensor = o1_embeds.unsqueeze(2).repeat(1, 1, num_query_o2, 1)
        o2_tensor = o2_embeds.unsqueeze(1).repeat(1, num_query_o1, 1, 1)

        relationship_tensor = torch.cat([o1_tensor, o2_tensor], dim=-1)
        relationship_pred = self.classifier(relationship_tensor)

        return relationship_pred

    def loss(self, rel_preds, o1_pos, o2_pos, o1_assign_results, o2_assign_results, gt_adjs):
        # rel_preds = rel_preds[-1]
        B, num_query_o1, num_query_o2, _ = rel_preds.size()
        o1_assign = o1_assign_results[-1]
        o1_pos_inds = o1_assign['pos_inds']
        o1_pos_assigned_gt_inds = o1_assign['pos_assigned_gt_inds']

        if self.shared_param:
            o2_assign = o1_assign
            o2_pos_inds = o1_pos_inds
            o2_pos_assigned_gt_inds = o1_pos_assigned_gt_inds
        else:
            o2_assign = o2_assign_results[-1]
            o2_pos_inds = o2_assign['pos_inds']
            o2_pos_assigned_gt_inds = o2_assign['pos_assigned_gt_inds']

        targets = []
        for i in range(B):
            gt_adj = gt_adjs[i]
            target = torch.zeros_like(rel_preds[i].squeeze(-1), dtype=gt_adj.dtype, device=rel_preds.device)
            xs = o1_pos_inds[i].unsqueeze(-1).repeat(1, o2_pos_inds[i].size(0))
            ys = o2_pos_inds[i].unsqueeze(0).repeat(o1_pos_inds[i].size(0), 1)
            target[xs, ys] = gt_adj[o1_pos_assigned_gt_inds[i]][:, o2_pos_assigned_gt_inds[i]]
            xs_new = o1_pos_inds[i]
            ys_new = o2_pos_inds[i]
            # target[xs_new, :][:, ys_new] = gt_adj[o1_pos_assigned_gt_inds[i]][:, o2_pos_assigned_gt_inds[i]]
            targets.append(target)
        targets = torch.stack(targets, dim=0)

        targets_copy = targets
        targets = 1 - targets[0][xs_new][:, ys_new].view(-1).long()
        rel_preds = rel_preds[0][xs_new][:, ys_new].view(-1, 1)

        loss_rel = self.loss_rel(rel_preds, targets)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_rel = torch.nan_to_num(loss_rel)

        if self.loss_ll_l1_weight == 0:
            return dict(loss_rel=loss_rel)
        else:
            loss_ll_l1 = self.loss_ll_l1(o1_pos, o2_pos, xs_new, ys_new, targets_copy)
            if digit_version(TORCH_VERSION) >= digit_version('1.8'):
                loss_ll_l1 = torch.nan_to_num(loss_ll_l1)

            return dict(loss_rel=loss_rel, loss_ll_l1=loss_ll_l1)

    def loss_ll_l1(self, o1_pos, o2_pos, xs_new, ys_new, targets):

        o1_pos = o1_pos[-1]
        o2_pos = o2_pos[-1]

        o1 = o1_pos[0].clone()
        o2 = o2_pos[0].clone()

        o1 = self.control_points_to_lane_points(o1)
        o2 = self.control_points_to_lane_points(o2)

        o1 = o1[..., -3:].unsqueeze(1).repeat(1, o1.size(0), 1)
        o2 = o2[..., :3].unsqueeze(0).repeat(o2.size(0), 1, 1)

        loss_dense = (torch.abs((o1 - o2)).mean(-1)[xs_new][:, ys_new] * targets[0][xs_new][:, ys_new]).mean()

        loss_dense = self.loss_ll_l1_weight * loss_dense

        return loss_dense

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