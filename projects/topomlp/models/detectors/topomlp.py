# ==============================================================================
# This module is for the implementation of TopoMLP.
# ==============================================================================

import torch

from mmdet3d.models import DETECTORS, build_neck, build_head
from mmdet3d.models.detectors import MVXTwoStageDetector

from mmdet.core import bbox_cxcywh_to_xyxy
import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16
from projects.topomlp.models.utils.grid_mask import GridMask


@DETECTORS.register_module()
class TopoMLP(MVXTwoStageDetector):

    def __init__(self,
                 use_grid_mask=False,
                 img_backbone=None,
                 img_neck=None,
                 lc_head=None,
                 te_head=None,
                 lclc_head=None,
                 lcte_head=None,
                 **kwargs):

        super().__init__(img_backbone=img_backbone, img_neck=img_neck, **kwargs)

        self.lc_head = build_head(lc_head)
        self.te_head = build_head(te_head)
        self.lclc_head = build_head(lclc_head)
        self.lcte_head = build_head(lcte_head)

        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.8, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img):
        """Extract features of images."""
        B, N, C, imH, imW = img.shape
        img = img.view(B * N, C, imH, imW)
        if self.use_grid_mask:
            img = self.grid_mask(img)
        x = self.img_backbone(img)
        if self.with_img_neck:
            x = self.img_neck(x)
        else:
            x = x[0]

        return x

    @force_fp32(apply_to=('img'))
    def simple_forward(self, img, img_metas):

        # extract image features
        B, N, C, imH, imW = img.shape
        img_feats = self.extract_img_feat(img)

        if len(img_feats) == 4:
            x = img_feats[2]
            _, output_dim, ouput_H, output_W = x.shape
            x = x.view(B, N, output_dim, ouput_H, output_W)
            pv_feat = []
            for img_feat in img_feats:
                _, output_dim, ouput_H, output_W = img_feat.shape
                img_feat = img_feat.view(B, N, output_dim, ouput_H, output_W)
                pv_feat.append(img_feat[:, 0, ...])
            # pv_feat = [lvl[:1] for lvl in img_feats]
        else:
            x = img_feats
            _, output_dim, ouput_H, output_W = x.shape
            x = x.view(B, N, output_dim, ouput_H, output_W)
            pv_feat = x[:, 0, ...]

        # lc
        lc_img_metas = [{
            'batch_input_shape': (x.shape[-2], x.shape[-1]),
            'img_shape': (x.shape[-2], x.shape[-1], None),
            'scale_factor': None,  # dummy
        } for _ in range(B)]
        all_lc_cls_scores_list, all_lc_preds_list, lc_outs_dec_list,\
         all_lc_cls_scores_one2many_list, all_lc_preds_one2many_list, lc_outs_dec_one2many_list = self.lc_head(x, img_metas)

        # te
        te_img_metas = [{
            'batch_input_shape': (img_metas[b]['pad_shape'][0][0], img_metas[b]['pad_shape'][0][1]),
            'img_shape': (img_metas[b]['img_shape'][0][0], img_metas[b]['img_shape'][0][1], None),
            'scale_factor': img_metas[b]['scale_factor'],
            'te_yolov8': img_metas[b]['te_yolov8'],
        } for b in range(B)]
        all_te_cls_scores_list, all_te_preds_list, te_outs_dec_list = self.te_head(pv_feat, te_img_metas) # all_te_cls_scores_list: D*B*Q*13, all_te_preds_list: D*B*Q*4, te_outs_dec_list: D*B*Q*256,

        # topology_lclc
        all_lclc_preds_list = self.lclc_head(
            lc_outs_dec_list,
            lc_outs_dec_list,
            all_lc_preds_list,
            all_lc_preds_list,
        )

        # topology_lcte
        all_lcte_preds_list = self.lcte_head(
            lc_outs_dec_list,
            te_outs_dec_list,
            img_metas
        )

        return {
            'all_lc_cls_scores_list': all_lc_cls_scores_list,
            'all_lc_preds_list': all_lc_preds_list,
            'lc_img_metas': lc_img_metas,
            'all_te_cls_scores_list': all_te_cls_scores_list,
            'all_te_preds_list': all_te_preds_list,
            'te_img_metas': te_img_metas,
            'all_lclc_preds_list': all_lclc_preds_list,
            'all_lcte_preds_list': all_lcte_preds_list,
            'all_lc_cls_scores_one2many_list': all_lc_cls_scores_one2many_list,
            'all_lc_preds_one2many_list': all_lc_preds_one2many_list,
            'lc_outs_dec_one2many_list': lc_outs_dec_one2many_list
        }

    def forward_train(self,
                      img,
                      img_metas,
                      gt_lc=None,
                      gt_lc_labels=None,
                      gt_te=None,
                      gt_te_labels=None,
                      gt_topology_lclc=None,
                      gt_topology_lcte=None,
                      **kwargs):

        outs = self.simple_forward(img, img_metas)

        losses = dict()

        # lc
        lc_loss_dict, lc_assign_results = self.lc_head.loss(
            outs['all_lc_cls_scores_list'],
            outs['all_lc_preds_list'],
            outs['all_lc_cls_scores_one2many_list'],
            outs['all_lc_preds_one2many_list'],
            gt_lc,
            gt_lc_labels,
            outs['lc_img_metas'],
        )
        losses.update({
            f'lc_{key}': val for key, val in lc_loss_dict.items()
        })

        # te
        te_loss_dict, te_assign_results = self.te_head.loss(
            outs['all_te_cls_scores_list'],
            outs['all_te_preds_list'],
            gt_te,
            gt_te_labels,
            outs['te_img_metas'],
        )
        losses.update({
            f'te_{key}': val for key, val in te_loss_dict.items()
        })

        # topology_lclc
        topology_lclc_loss_dict = self.lclc_head.loss(
            outs['all_lclc_preds_list'],
            outs['all_lc_preds_list'],
            outs['all_lc_preds_list'],
            lc_assign_results,
            lc_assign_results,
            gt_topology_lclc,
        )
        losses.update({
            f'topology_lclc_{key}': val for key, val in topology_lclc_loss_dict.items()
        })

        # topology_lcte
        topology_lcte_loss_dict = self.lcte_head.loss(
            outs['all_lcte_preds_list'],
            lc_assign_results,
            te_assign_results,
            gt_topology_lcte
        )
        losses.update({
            f'topology_lcte_{key}': val for key, val in topology_lcte_loss_dict.items()
        })

        return losses

    def forward_test(self, img, img_metas, **kwargs):

        outs = self.simple_forward(img, img_metas)

        index_lc = torch.sigmoid(outs['all_lc_cls_scores_list'][-1]) > 0.0
        index_lc = index_lc.squeeze()
        outs['all_lc_cls_scores_list'] = outs['all_lc_cls_scores_list'][:, :, index_lc, :]
        outs['all_lc_preds_list'] = outs['all_lc_preds_list'][:, :, index_lc, :]
        outs['all_lclc_preds_list'] = outs['all_lclc_preds_list'][:, index_lc][:,:,index_lc]
        outs['all_lcte_preds_list'] = outs['all_lcte_preds_list'][:, index_lc]

        # all_te_cls_scores_list = outs['all_te_cls_scores_list'][-1].max(-1)[0]
        # index_te = torch.sigmoid(all_te_cls_scores_list) > 0.1
        # index_te = index_te.squeeze()
        # outs['all_te_cls_scores_list'] = outs['all_te_cls_scores_list'][:, :, index_te, :]
        # outs['all_te_preds_list'] = outs['all_te_preds_list'][:, :, index_te, :]
        # outs['all_lcte_preds_list'] = outs['all_lcte_preds_list'][:, index_lc][:, :, index_te]

        pred_lc = self.lc_head.get_bboxes(
            outs['all_lc_cls_scores_list'],
            outs['all_lc_preds_list'],
            outs['lc_img_metas'],
        )
        pred_te = self.te_head.get_bboxes(
            outs['all_te_cls_scores_list'],
            outs['all_te_preds_list'],
            outs['te_img_metas'],
            rescale=True,
        )

        pred_topology_lclc = self.lclc_head.get_topology(outs['all_lclc_preds_list'])

        pred_topology_lcte = self.lcte_head.get_topology(outs['all_lcte_preds_list'])

        assert len(pred_lc) == len(pred_te) == 1, \
            'evaluation implemented for bs=1'
        return [{
            'pred_lc': pred_lc[0],
            'pred_te': pred_te[0],
            'pred_topology_lclc': pred_topology_lclc[0],
            'pred_topology_lcte': pred_topology_lcte[0],
        }]

