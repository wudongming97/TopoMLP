# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# transforms.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
#
# Contact wanghuijie@pjlab.org.cn if you have any issue.
#
# Copyright (c) 2023 The OpenLane-v2 Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import copy

import numpy as np
from numpy import random
from math import factorial
import torch
from PIL import Image


import mmcv
from mmdet.datasets import PIPELINES

import cv2
from openlanev2.visualization.utils import COLOR_DICT
from projects.topomlp.models.utils.grid_mask import GridMask


COLOR_GT = (0, 255, 0)
COLOR_GT_TOPOLOGY = (0, 127, 0)
COLOR_PRED = (0, 0, 255)
COLOR_PRED_TOPOLOGY = (0, 0, 127)
COLOR_DICT = {k: (v[2], v[1], v[0]) for k, v in COLOR_DICT.items()}


def render_pv(images, lidar2imgs, gt_lc, pred_lc, gt_te, gt_te_attr, pred_te, pred_te_attr):
    results = []
    scale = 30

    for idx, (image, lidar2img) in enumerate(zip(images, lidar2imgs)):
        # print(idx)

        if gt_lc is not None:
            for lc_i, lc in enumerate(gt_lc):
                # if lc_i == 38 :
                #     print(lc_i)
                xyz1 = np.concatenate([lc, np.ones((lc.shape[0], 1))], axis=1)
                xyz1 = xyz1 @ lidar2img.T
                xyz1 = xyz1[xyz1[:, 2] > 1e-5]
                if xyz1.shape[0] == 0:
                    continue
                points_2d = xyz1[:, :2] / xyz1[:, 2:3]

                points_2d = np.clip(points_2d, -10000, 10000)
                points_2d = points_2d.astype(int)
                image = cv2.circle(image, (int(points_2d[0, 0]), int(points_2d[0, 1])),
                                   max(round(scale * 0.5), 3), COLOR_PRED, -1)
                image = cv2.polylines(image, points_2d[None], False, COLOR_PRED, 10)
                # if points_2d[0, 0] is not None and points_2d[0, 1] is not None:
                #     # print(points_2d[0])
                #     print(image.dtype)

                # image = cv2.circle(image, (points_2d[-1, 0], points_2d[-1, 1]), max(round(scale * 0.5), 3), COLOR_PRED,
                #                    -1)

        if pred_lc is not None:
            for lc in pred_lc:
                xyz1 = np.concatenate([lc, np.ones((lc.shape[0], 1))], axis=1)
                xyz1 = xyz1 @ lidar2img.T
                xyz1 = xyz1[xyz1[:, 2] > 1e-5]
                if xyz1.shape[0] == 0:
                    continue
                points_2d = xyz1[:, :2] / xyz1[:, 2:3]

                points_2d = points_2d.astype(int)
                image = cv2.polylines(image, points_2d[None], False, COLOR_PRED, 2)

        if idx == 0:  # front view image

            if gt_te is not None:
                for bbox, attr in zip(gt_te, gt_te_attr):
                    b = bbox.astype(np.int32)
                    image = render_corner_rectangle(image, (b[0], b[1]), (b[2], b[3]), COLOR_DICT[attr], 3, 1)

            if pred_te is not None:
                for bbox, attr in zip(pred_te, pred_te_attr):
                    b = bbox.astype(np.int32)
                    image = cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), COLOR_DICT[attr], 3)

        results.append(image)

    return results


def render_corner_rectangle(img, pt1, pt2, color,
                            corner_thickness=3, edge_thickness=2,
                            centre_cross=False, lineType=cv2.LINE_8):
    corner_length = min(abs(pt1[0] - pt2[0]), abs(pt1[1] - pt2[1])) // 4
    e_args = [color, edge_thickness, lineType]
    c_args = [color, corner_thickness, lineType]

    # edges
    img = cv2.line(img, (pt1[0] + corner_length, pt1[1]), (pt2[0] - corner_length, pt1[1]), *e_args)
    img = cv2.line(img, (pt2[0], pt1[1] + corner_length), (pt2[0], pt2[1] - corner_length), *e_args)
    img = cv2.line(img, (pt1[0], pt1[1] + corner_length), (pt1[0], pt2[1] - corner_length), *e_args)
    img = cv2.line(img, (pt1[0] + corner_length, pt2[1]), (pt2[0] - corner_length, pt2[1]), *e_args)

    # corners
    img = cv2.line(img, pt1, (pt1[0] + corner_length, pt1[1]), *c_args)
    img = cv2.line(img, pt1, (pt1[0], pt1[1] + corner_length), *c_args)
    img = cv2.line(img, (pt2[0], pt1[1]), (pt2[0] - corner_length, pt1[1]), *c_args)
    img = cv2.line(img, (pt2[0], pt1[1]), (pt2[0], pt1[1] + corner_length), *c_args)
    img = cv2.line(img, (pt1[0], pt2[1]), (pt1[0] + corner_length, pt2[1]), *c_args)
    img = cv2.line(img, (pt1[0], pt2[1]), (pt1[0], pt2[1] - corner_length), *c_args)
    img = cv2.line(img, pt2, (pt2[0] - corner_length, pt2[1]), *c_args)
    img = cv2.line(img, pt2, (pt2[0], pt2[1] - corner_length), *c_args)

    if centre_cross:
        cx, cy = int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2)
        img = cv2.line(img, (cx - corner_length, cy), (cx + corner_length, cy), *e_args)
        img = cv2.line(img, (cx, cy - corner_length), (cx, cy + corner_length), *e_args)

    return img


def render_bev(gt_lc=None, pred_lc=None, gt_topology_lclc=None, pred_topology_lclc=None, map_size=[-52, 52, -27, 27],
               scale=20, confidence=None):
    image = np.zeros((int(scale * (map_size[1] - map_size[0])), int(scale * (map_size[3] - map_size[2])), 3),
                     dtype=np.uint8)

    if gt_lc is not None:
        for lc in gt_lc:
            draw_coor = (scale * (-lc[:, :2] + np.array([map_size[1], map_size[3]]))).astype(np.int)
            image = cv2.polylines(image, [draw_coor[:, [1, 0]]], False, COLOR_GT, max(round(scale * 0.2), 1))
            image = cv2.circle(image, (draw_coor[0, 1], draw_coor[0, 0]), max(round(scale * 0.5), 3), COLOR_GT, -1)
            # image = cv2.circle(image, (draw_coor[-1, 1], draw_coor[-1, 0]), max(round(scale * 0.5), 3), COLOR_GT, -1)

    if gt_topology_lclc is not None:
        for l1_idx, lclc in enumerate(gt_topology_lclc):
            for l2_idx, connected in enumerate(lclc):
                if connected:
                    l1 = gt_lc[l1_idx]
                    l2 = gt_lc[l2_idx]
                    l1_mid = len(l1) // 2
                    l2_mid = len(l2) // 2
                    p1 = (scale * (-l1[l1_mid, :2] + np.array([map_size[1], map_size[3]]))).astype(np.int)
                    p2 = (scale * (-l2[l2_mid, :2] + np.array([map_size[1], map_size[3]]))).astype(np.int)
                    image = cv2.arrowedLine(image, (p1[1], p1[0]), (p2[1], p2[0]), COLOR_GT_TOPOLOGY,
                                            max(round(scale * 0.1), 1), tipLength=0.03)

    if pred_lc is not None:
        if confidence is not None:
            for lc in pred_lc:
                draw_coor = (scale * (-lc[:, :2] + np.array([map_size[1], map_size[3]]))).astype(np.int)
                image = cv2.polylines(image, [draw_coor[:, [1, 0]]], False, COLOR_PRED, max(round(scale * 0.2), 1))
                image = cv2.circle(image, (draw_coor[0, 1], draw_coor[0, 0]), max(round(scale * 0.5), 3), COLOR_PRED,
                                   -1)
                # image = cv2.circle(image, (draw_coor[-1, 1], draw_coor[-1, 0]), max(round(scale * 0.5), 3), COLOR_PRED, -1)
        else:
            for i, lc in enumerate(pred_lc):
                draw_coor = (scale * (-lc[:, :2] + np.array([map_size[1], map_size[3]]))).astype(np.int)
                image = cv2.polylines(image, [draw_coor[:, [1, 0]]], False, COLOR_PRED, max(round(scale * 0.2), 1))
                image = cv2.circle(image, (draw_coor[0, 1], draw_coor[0, 0]), max(round(scale * 0.5), 3), COLOR_PRED,
                                   -1)
                image = cv2.putText(image, str(confidence[i]), (draw_coor[0, 1], draw_coor[0, 0]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_PRED, 2)

    if pred_topology_lclc is not None:
        for l1_idx, lclc in enumerate(pred_topology_lclc):
            for l2_idx, connected in enumerate(lclc):
                if connected:
                    l1 = pred_lc[l1_idx]
                    l2 = pred_lc[l2_idx]
                    l1_mid = len(l1) // 2
                    l2_mid = len(l2) // 2
                    p1 = (scale * (-l1[l1_mid, :2] + np.array([map_size[1], map_size[3]]))).astype(np.int)
                    p2 = (scale * (-l2[l2_mid, :2] + np.array([map_size[1], map_size[3]]))).astype(np.int)
                    image = cv2.arrowedLine(image, (p1[1], p1[0]), (p2[1], p2[0]), COLOR_PRED_TOPOLOGY,
                                            max(round(scale * 0.1), 1), tipLength=0.03)

    return image


@PIPELINES.register_module()
class PlotLaneOnImages():
    def __init__(self):
        pass

    def __call__(self, results):
        assert 'ring_front_center' in results['img_paths'][0], \
            'the first image should be the front view'

        images = results['img']
        # print(results['scene_token'])
        # print(results['sample_idx'])
        # if results['scene_token'] == '00283' and results['sample_idx'] == '315972299149927220':
        #     print(results['scene_token'])
        images = render_pv(
            images, results['lidar2img'],
            gt_lc=results['gt_lc'], pred_lc=None,
            gt_te=results['gt_te'], gt_te_attr=results['gt_te_labels'], pred_te=None, pred_te_attr=None,
        )
        for i in range(7):
            mmcv.imwrite(images[i], '/data/wudongming/mmdetection3d/images/{}.jpg'.format(str(i)))
        # results['img'] = images

        bev_lane = render_bev(
            gt_lc=results['gt_lc'], pred_lc=None,
            map_size=[-52, 55, -27, 27], scale=20,
        )
        mmcv.imwrite(bev_lane, '/data/wudongming/mmdetection3d/images/bev.jpg')

        # return results
        return

@PIPELINES.register_module()
class GridMaskImage():
    def __init__(self, use_grid_mask=False, ratio=0.5):
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=ratio, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask

    def __call__(self, results):
        assert 'ring_front_center' in results['img_paths'][0], \
            'the first image should be the front view'

        if self.use_grid_mask:
            img = results['img']
            B, N, C, imH, imW = img.shape
            img = img.view(B * N, C, imH, imW)
            img = self.grid_mask(img)
            img = img.view(B, N, C, imH, imW)
            results['img'] = img

        # return results
        return results

@PIPELINES.register_module()
class AV2ResizeCropFlipRotImage():
    def __init__(self, data_aug_conf=None, training=True, multi_stamps=False):
        self.data_aug_conf = data_aug_conf
        self.training = training
        self.min_size = 2.0
        self.multi_stamps = multi_stamps

        self.front_img_rot90 = ('front_img_rot90' in data_aug_conf) and data_aug_conf['front_img_rot90']

    def __call__(self, results):
        # results_ori = copy.deepcopy(results)
        imgs = results['img']
        N = len(imgs) // 2 if self.multi_stamps else len(imgs)
        new_imgs = []
        # new_gt_bboxes = []
        # new_centers2d = []
        # new_gt_labels = []
        # new_depths = []

        assert self.data_aug_conf['rot_lim'] == (0.0, 0.0), "Rotation is not currently supported"

        for i in range(N):
            resize, resize_dims, crop, flip, rotate = self._sample_augmentation(imgs[i])
            img = Image.fromarray(np.uint8(imgs[i]))
            img, ida_mat, h_min, w_min = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            # if self.training:
            #     gt_bboxes = results['gt_bboxes'][i]
            #     centers2d = results['centers2d'][i]
            #     gt_labels = results['gt_labels'][i]
            #     depths = results['depths'][i]
            #     if len(gt_bboxes) != 0:
            #         gt_bboxes, centers2d, gt_labels, depths = self._bboxes_transform(
            #             imgs[i],
            #             gt_bboxes,
            #             centers2d,
            #             gt_labels,
            #             depths,
            #             resize=resize,
            #             crop=crop,
            #             flip=flip,
            #         )
            #     if len(gt_bboxes) != 0:
            #         gt_bboxes, centers2d, gt_labels, depths = self._filter_invisible(
            #             imgs[i],
            #             gt_bboxes,
            #             centers2d,
            #             gt_labels,
            #             depths
            #         )
            #
            #     new_gt_bboxes.append(gt_bboxes)
            #     new_centers2d.append(centers2d)
            #     new_gt_labels.append(gt_labels)
            #     new_depths.append(depths)

            new_imgs.append(np.array(img).astype(np.float32))
            results['cam_intrinsic'][i][:3, :3] = ida_mat @ results['cam_intrinsic'][i][:3, :3]
            results['img_shape'][i] = (h_min, w_min, 3)

            if i == 0:
                # gt
                scale_factor = np.array(
                    [resize, resize, resize, resize],
                    dtype=np.float32,
                )
                results['scale_factor'] = scale_factor
                if 'gt_te' in results:
                    results['gt_te'] = results['gt_te'] * results['scale_factor']
        # results['gt_bboxes'] = new_gt_bboxes
        # results['centers2d'] = new_centers2d
        # results['gt_labels'] = new_gt_labels
        # results['depths'] = new_depths
        results['img'] = new_imgs
        results['cam2imgs'] = [results['cam_intrinsic'][i][:3, :3] for i in
                                range(len(results['lidar2cam']))]
        results['lidar2img'] = [results['cam_intrinsic'][i] @ results['lidar2cam'][i] for i in
                                range(len(results['lidar2cam']))]

        # results['img_shape'] = [img.shape for img in new_imgs]
        results['pad_shape'] = [img.shape for img in new_imgs]

        return results

    def _bboxes_transform(self, img, bboxes, centers2d, gt_labels, depths, resize, crop, flip):
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        H, W = img.shape[:2]
        if (H > W) and self.front_img_rot90:
            # resize, crop, rot90
            return self._bboxes_transform_rot90(img, bboxes, centers2d, gt_labels, depths, resize, crop, flip)

        if H > W:
            fH, fW = self.data_aug_conf["final_dim_f"]
        else:
            fH, fW = self.data_aug_conf["final_dim"]
        bboxes = bboxes * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        keep = ((bboxes[:, 2] - bboxes[:, 0]) >= self.min_size) & ((bboxes[:, 3] - bboxes[:, 1]) >= self.min_size)

        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1
        # normalize
        bboxes = bboxes[keep]

        centers2d = centers2d * resize
        centers2d[:, 0] = centers2d[:, 0] - crop[0]
        centers2d[:, 1] = centers2d[:, 1] - crop[1]
        centers2d[:, 0] = np.clip(centers2d[:, 0], 0, fW)
        centers2d[:, 1] = np.clip(centers2d[:, 1], 0, fH)
        if flip:
            centers2d[:, 0] = fW - centers2d[:, 0]
        # normalize

        centers2d = centers2d[keep]
        gt_labels = gt_labels[keep]
        depths = depths[keep]

        return bboxes, centers2d, gt_labels, depths

    def _filter_invisible(self, img, bboxes, centers2d, gt_labels, depths):
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        H, W = img.shape[:2]
        if (H > W) and self.front_img_rot90:
            fH, fW = self.data_aug_conf["final_dim"]
        else:
            if H > W:
                fH, fW = self.data_aug_conf["final_dim_f"]
            else:
                fH, fW = self.data_aug_conf["final_dim"]

        indices_maps = np.zeros((fH, fW))
        tmp_bboxes = np.zeros_like(bboxes)
        tmp_bboxes[:, :2] = np.ceil(bboxes[:, :2])
        tmp_bboxes[:, 2:] = np.floor(bboxes[:, 2:])
        tmp_bboxes = tmp_bboxes.astype(np.int64)
        sort_idx = np.argsort(-depths, axis=0, kind='stable')
        tmp_bboxes = tmp_bboxes[sort_idx]
        bboxes = bboxes[sort_idx]
        depths = depths[sort_idx]
        centers2d = centers2d[sort_idx]
        gt_labels = gt_labels[sort_idx]
        for i in range(bboxes.shape[0]):
            u1, v1, u2, v2 = tmp_bboxes[i]
            indices_maps[v1:v2, u1:u2] = i
        indices_res = np.unique(indices_maps).astype(np.int64)
        bboxes = bboxes[indices_res]
        depths = depths[indices_res]
        centers2d = centers2d[indices_res]
        gt_labels = gt_labels[indices_res]

        return bboxes, centers2d, gt_labels, depths

    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):

        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        h_1, w_1 = np.array(img).shape[:2]
        img = img.crop(crop)
        h_2, w_2 = np.array(img).shape[:2]
        h_min = min(h_1, h_2)
        w_min = min(w_1, w_2)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran

        # Optional: rot90 transformation
        H, W = np.array(img).shape[:2]
        if (H > W) and self.front_img_rot90:
            img = img.transpose(Image.ROTATE_90)
            _R = np.array(
                [[0, 1, 0],
                 [-1, 0, W - 1],
                 [0, 0, 1], ]
            )
            ida_mat = np.matmul(_R, ida_mat)

        return img, ida_mat, h_min, w_min

    def _sample_augmentation(self, img):
        H, W = img.shape[:2]
        if (H > W) and self.front_img_rot90:
            # resize, crop, rot90
            return self._sample_augmentation_rot90(img)

        if H > W:  # different front image size, need padding later
            fH, fW = self.data_aug_conf["final_dim_f"]
        else:
            fH, fW = self.data_aug_conf["final_dim"]
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = True
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def _sample_augmentation_rot90(self, img):
        '''
        resize, crop, rot90
        Return an image with size (W=final_H, H=final_W)
        '''
        H, W = img.shape[:2]
        fH, fW = self.data_aug_conf["final_dim"]
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims

            # this crop contrary to normal， H and W exchanged.
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fW
            crop_w = int(np.random.uniform(0, max(0, newW - fH)))
            crop = (crop_w, crop_h, crop_w + fH, crop_h + fW)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fW / H, fH / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fW
            crop_w = int(max(0, newW - fH) / 2)
            crop = (crop_w, crop_h, crop_w + fH, crop_h + fW)
            flip = True
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def _bboxes_transform_rot90(self, img, bboxes, centers2d, gt_labels, depths, resize, crop, flip):
        fH, fW = self.data_aug_conf["final_dim"]  # box H_max < fW, box W_max < fH
        bboxes = bboxes * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fH)  # W 方向
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fH)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fW)  # H 方向
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fW)
        keep = ((bboxes[:, 2] - bboxes[:, 0]) >= self.min_size) & ((bboxes[:, 3] - bboxes[:, 1]) >= self.min_size)

        if flip:  # flip should fH-1-x
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fH - 1 - x0
            bboxes[:, 0] = fH - 1 - x1
        # normalize
        bboxes = bboxes[keep]

        centers2d = centers2d * resize
        centers2d[:, 0] = centers2d[:, 0] - crop[0]
        centers2d[:, 1] = centers2d[:, 1] - crop[1]
        centers2d[:, 0] = np.clip(centers2d[:, 0], 0, fH)  # W 方向
        centers2d[:, 1] = np.clip(centers2d[:, 1], 0, fW)  # H 方向
        if flip:
            centers2d[:, 0] = fH - 1 - centers2d[:, 0]
        # normalize

        # Newly add: rot 90 transformation (x, y) -> (W-1-y, x); for (w, h) style, (y, x) -> (x, W-1-y)
        tmp = bboxes[:, 0].copy()  # y
        bboxes[:, 0] = bboxes[:, 1]  # x
        bboxes[:, 1] = fH - 1 - tmp
        tmp2 = bboxes[:, 2].copy()  # y
        bboxes[:, 2] = bboxes[:, 3]  # x
        bboxes[:, 3] = fH - 1 - tmp2

        t1y, t1x, t2y, t2x = bboxes[:, 0].copy(), bboxes[:, 1].copy(), bboxes[:, 2].copy(), bboxes[:, 3].copy()
        t4y, t4x, t3y, t3x = t1y, t2x, t2y, t1x
        bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3] = t4y, t4x, t3y, t3x
        # but original two coordinates are in the opposite direction

        tmp = centers2d[:, 0].copy()  # y
        centers2d[:, 0] = centers2d[:, 1]  # x
        centers2d[:, 1] = fH - 1 - tmp

        centers2d = centers2d[keep]
        gt_labels = gt_labels[keep]
        depths = depths[keep]

        return bboxes, centers2d, gt_labels, depths


@PIPELINES.register_module()
class AV2PadMultiViewImage():
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        assert size is not None or size_divisor is not None
        # assert size_divisor is None or size is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size == 'same2max':
            max_shape = max([img.shape for img in results['img']])[:2]
            padded_img = [mmcv.impad(img, shape=max_shape, pad_val=self.pad_val) for img in results['img']]
        elif self.size is not None:
            padded_img = [mmcv.impad(img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
        if self.size_divisor is not None:
            padded_img = [
                mmcv.impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val) for img in results['img']
            ]

        # results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        # results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class ResizeFrontView:

    def __init__(self):
        pass

    def __call__(self, results):
        assert 'ring_front_center' in results['img_paths'][0] or 'CAM_FRONT' in results['img_paths'][0], \
            'the first image should be the front view'

        #image
        front_view = results['img'][0]
        h, w, _ = front_view.shape
        resiezed_front_view, w_scale, h_scale = mmcv.imresize(
            front_view,
            (h, w),
            return_scale=True,
        )
        results['img'][0] = resiezed_front_view
        results['img_shape'][0] = resiezed_front_view.shape

        # gt
        scale_factor = np.array(
            [w_scale, h_scale, w_scale, h_scale],
            dtype=np.float32,
        )
        results['scale_factor'] = scale_factor
        if 'gt_te' in results:
            results['gt_te'] = results['gt_te'] * results['scale_factor']

        # intrinsic
        lidar2cam_r = results['rots'][0]
        lidar2cam_t = (-results['trans'][0]) @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t

        intrinsic = results['cam2imgs'][0]
        viewpad = np.eye(4)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

        cam_s = np.eye(4)
        cam_s[0, 0] *= w_scale
        cam_s[1, 1] *= h_scale

        viewpad = cam_s @ viewpad 
        intrinsic = viewpad[:intrinsic.shape[0], :intrinsic.shape[1]]
        lidar2img_rt = (viewpad @ lidar2cam_rt.T)

        results['cam_intrinsic'][0] = viewpad
        results['lidar2img'][0] = lidar2img_rt
        results['cam2imgs'][0] = intrinsic

        return results

@PIPELINES.register_module()
class CropFrontViewImageForAv2(object):

    def __init__(self, crop_h=(0, 1906)):
        self.crop_h = crop_h

    def _crop_img(self, results):
        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'][0] = results['img'][0][self.crop_h[0]:self.crop_h[1]]
        results['img_shape'] = [img.shape for img in results['img']]
        # results['crop_shape'][0] = np.array([0, self.crop_h[0]])

    def _crop_cam_intrinsic(self, results):
        results['cam_intrinsic'][0][1, 2] -= self.crop_h[0]
        results['lidar2img'][0] = results['cam_intrinsic'][0] @ results['lidar2cam'][0]

    def _crop_bbox(self, results):
        if 'gt_te' in results.keys():
            results['gt_te'][:, 1] -= self.crop_h[0]
            results['gt_te'][:, 3] -= self.crop_h[0]

            mask = results['gt_te'][:, 3] > 0
            results['gt_te'] = results['gt_te'][mask]
            results['gt_te'] = results['gt_te'][mask]
            if 'gt_topology_lcte' in results.keys():
                results['gt_topology_lcte'] = results['gt_topology_lcte'][:, mask]

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._crop_img(results)
        self._crop_cam_intrinsic(results)
        self._crop_bbox(results)
        return results

@PIPELINES.register_module()
class ResizeMultiview3D:
    """Resize images & bbox & mask.
    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.
    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:
    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.
    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.
        Args:
            img_scales (list[tuple]): Images scales for selection.
        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.
        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.
        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where \
                ``img_scale`` is sampled scale and None is just a placeholder \
                to be consistent with :func:`random_select`.
        """

        # assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        assert mmcv.is_list_of(img_scales, tuple)
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.
        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.
        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.
        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where \
                ``scale`` is sampled ratio multiplied with ``img_scale`` and \
                None is just a placeholder to be consistent with \
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.
        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.
        Args:
            results (dict): Result dict from :obj:`dataset`.
        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        # results['scale'] = (1280, 720)
        img_shapes = []
        pad_shapes = []
        scale_factors = []
        keep_ratios = []
        # import ipdb;ipdb.set_trace()
        for i in range(len(results['img'])):
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results['img'][i],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results['img'][i].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results['img'][i],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
            results['img'][i] = img
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            img_shapes.append(img.shape)
            pad_shapes.append(img.shape)
            scale_factors.append(scale_factor)
            keep_ratios.append(self.keep_ratio)
            # rescale the camera intrinsic
            results['cam_intrinsic'][i][0, 0] *= w_scale
            results['cam_intrinsic'][i][0, 2] *= w_scale
            results['cam_intrinsic'][i][1, 1] *= h_scale
            results['cam_intrinsic'][i][1, 2] *= h_scale
        if 'gt_te' in results:
            results['gt_te'] = results['gt_te'] * scale_factors[0]
        if 'scale_factor' in results.keys():
            results['scale_factor'] = results['scale_factor'] * scale_factors[0]
        else:
            results['scale_factor'] = scale_factors[0]
        results['img_shape'] = img_shapes
        results['pad_shape'] = pad_shapes
        # results['scale_factor'] = scale_factors[0]
        results['keep_ratio'] = keep_ratios

        results['lidar2img'] = [results['cam_intrinsic'][i] @ results['lidar2cam'][i] for i in
                                range(len(results['lidar2cam']))]

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                # if 'scale_factor' in results:
                #     results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage:
    r"""
    Notes
    -----
    Adapted from https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py#L62.

    Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb


    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results['img'] = [mmcv.imnormalize(img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str

@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    r"""
    Notes
    -----
    Adapted from https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py#L99.
    
    Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str

@PIPELINES.register_module()
class CustomPadMultiViewImage:

    def __init__(self, size_divisor=None, pad_val=0):
        self.size_divisor = size_divisor
        self.pad_val = pad_val

    def __call__(self, results):
        max_h = max([img.shape[0] for img in results['img']])
        max_w = max([img.shape[1] for img in results['img']])
        padded_img = [mmcv.impad(img, shape=(max_h, max_w), pad_val=self.pad_val) for img in results['img']]
        if self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in padded_img]
        
        results['img'] = padded_img
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = None
        results['pad_size_divisor'] = self.size_divisor

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str

@PIPELINES.register_module()
class CustomParameterizeLane:

    def __init__(self, method, method_para):
        method_list = ['bezier', 'polygon', 'bezier_Direction_attribute', 'bezier_Endpointfixed']
        self.method = method
        if not self.method in method_list:
            raise Exception("Not implemented!")
        self.method_para = method_para

    def __call__(self, results):
        centerlines = results['gt_lc']
        para_centerlines = getattr(self, self.method)(centerlines, **self.method_para)
        results['gt_lc'] = para_centerlines
        return results

    def comb(self, n, k):
        return factorial(n) // (factorial(k) * factorial(n - k))

    def fit_bezier(self, points, n_control):
        n_points = len(points)
        A = np.zeros((n_points, n_control))
        t = np.arange(n_points) / (n_points - 1)
        for i in range(n_points):
            for j in range(n_control):
                A[i, j] = self.comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
        conts = np.linalg.lstsq(A, points, rcond=None)
        return conts

    def fit_bezier_Endpointfixed(self, points, n_control):
        n_points = len(points)
        A = np.zeros((n_points, n_control))
        t = np.arange(n_points) / (n_points - 1)
        for i in range(n_points):
            for j in range(n_control):
                A[i, j] = self.comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
        A_BE = A[1:-1, 1:-1]
        _points = points[1:-1]
        _points = _points - A[1:-1, 0].reshape(-1, 1) @ points[0].reshape(1, -1) - A[1:-1, -1].reshape(-1, 1) @ points[-1].reshape(1, -1)

        conts = np.linalg.lstsq(A_BE, _points, rcond=None)

        control_points = np.zeros((n_control, points.shape[1]))
        control_points[0] = points[0]
        control_points[-1] = points[-1]
        control_points[1:-1] = conts[0]

        return control_points

    def bezier(self, input_data, n_control=2):

        coeffs_list = []
        for idx, centerline in enumerate(input_data):
            sorted_x = np.array(centerline[:, 1])
            sorted_y = np.array(centerline[:, 0])
            points = np.array(list(zip(sorted_x, sorted_y)))
            res = self.fit_bezier(points, n_control)[0]
            start_res = res[0]
            end_res = res[-1]
            first_diff = (np.sum(np.square(start_res - points[0]))) + (np.sum(np.square(end_res - points[-1])))
            second_diff = (np.sum(np.square(start_res - points[-1]))) + (np.sum(np.square(end_res - points[0])))

            if first_diff <= second_diff:
                fin_res = res
            else:
                fin_res = np.zeros_like(res)
                for m in range(len(res)):
                    fin_res[len(res) - m - 1] = res[m]

            fin_res = np.clip(fin_res, 0, 1)
            coeffs_list.append(np.reshape(np.float32(fin_res), (-1)))

        return np.array(coeffs_list)

    def bezier_Direction_attribute(self, input_data, n_control=3):
        coeffs_list = []
        for idx, centerline in enumerate(input_data):
            centerline[:, 1] = centerline[:, 1]
            centerline[:, 0] = centerline[:, 0]
            sorted_x = np.array(centerline[:, 1])
            sorted_y = np.array(centerline[:, 0])
            points = np.array(list(zip(sorted_x, sorted_y)))
            res = self.fit_bezier(points, n_control)[0]
            fin_res = np.clip(res, 0, 1)
            start_res = res[0]
            end_res = res[-1]
            first_diff = (np.sum(np.square(start_res - points[0]))) + (np.sum(np.square(end_res - points[-1])))
            second_diff = (np.sum(np.square(start_res - points[-1]))) + (np.sum(np.square(end_res - points[0])))
            if first_diff <= second_diff:
                da = 0
            else:
                da = 1
            fin_res = np.append(fin_res, da)
            coeffs_list.append(np.reshape(np.float32(fin_res), (-1)))
        return np.array(coeffs_list)

    def bezier_Endpointfixed(self, input_data, n_control=2):
        coeffs_list = []
        for idx, centerline in enumerate(input_data):
            res = self.fit_bezier_Endpointfixed(centerline, n_control)
            coeffs = res.flatten()
            coeffs_list.append(coeffs)
        return np.array(coeffs_list, dtype=np.float32)

    def polygon(self, input_data, key_rep='Bounding Box'):
        keypoints = []
        for idx, centerline in enumerate(input_data):
            centerline[:, 1] = centerline[:, 1]
            centerline[:, 0] = centerline[:, 0]
            sorted_x = np.array(centerline[:, 1])
            sorted_y = np.array(centerline[:, 0])
            points = np.array(list(zip(sorted_x, sorted_y)))
            if key_rep not in ['Bounding Box', 'SME', 'Extreme Points']:
                raise Exception(f"{key_rep} not existed!")
            elif key_rep == 'Bounding Box':
                res = np.array(
                    [points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()]).reshape((2, 2))
                keypoints.append(np.reshape(np.float32(res), (-1)))
            elif key_rep == 'SME':
                res = np.array([points[0], points[-1], points[int(len(points) / 2)]])
                keypoints.append(np.reshape(np.float32(res), (-1)))
            else:
                min_x = np.min([points[:, 0] for p in points])
                ind_left = np.where(points[:, 0] == min_x)
                max_x = np.max([points[:, 0] for p in points])
                ind_right = np.where(points[:, 0] == max_x)
                max_y = np.max([points[:, 1] for p in points])
                ind_top = np.where(points[:, 1] == max_y)
                min_y = np.min([points[:, 1] for p in points])
                ind_botton = np.where(points[:, 1] == min_y)
                res = np.array(
                    [points[ind_left[0][0]], points[ind_right[0][0]], points[ind_top[0][0]], points[ind_botton[0][0]]])
                keypoints.append(np.reshape(np.float32(res), (-1)))
        return np.array(keypoints)


@PIPELINES.register_module()
class GlobalRotScaleTransImage(object):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(
        self,
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        # reverse_angle=False,
        training=True,
    ):

        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

        # self.reverse_angle = reverse_angle
        self.training = training

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        # random rotate
        rot_angle = np.random.uniform(*self.rot_range)

        self.rotate_bev_along_z(results, rot_angle)
        rot_cos = torch.cos(torch.tensor(rot_angle))
        rot_sin = torch.sin(torch.tensor(rot_angle))
        rot_mat = torch.tensor([[rot_cos, -rot_sin,0], [rot_sin, rot_cos,0],[0,0,1]])
        # import ipdb;ipdb.set_trace()
        results['gt_lc'] = (torch.tensor(results['gt_lc']) @ rot_mat.T).numpy()
        # if self.reverse_angle:
        #     rot_angle *= -1
        # results["gt_bboxes_3d"].rotate(
        #     np.array(rot_angle)
        # )  # mmdet LiDARInstance3DBoxes存的角度方向是反的(rotate函数实现的是绕着z轴由y向x转)

        # random scale
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        self.scale_xyz(results, scale_ratio)
        # results["gt_bboxes_3d"].scale(scale_ratio)

        # TODO: support translation

        return results

    def rotate_bev_along_z(self, results, angle):
        rot_cos = torch.cos(torch.tensor(angle))
        rot_sin = torch.sin(torch.tensor(angle))

        rot_mat = torch.tensor([[rot_cos, -rot_sin, 0, 0], [rot_sin, rot_cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        rot_mat_inv = torch.inverse(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ rot_mat_inv).numpy()
            # results["extrinsics"][view] = (torch.tensor(results["extrinsics"][view]).float() @ rot_mat_inv).numpy()

        return

    def scale_xyz(self, results, scale_ratio):
        rot_mat = torch.tensor(
            [
                [scale_ratio, 0, 0, 0],
                [0, scale_ratio, 0, 0],
                [0, 0, scale_ratio, 0],
                [0, 0, 0, 1],
            ]
        )

        rot_mat_inv = torch.inverse(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ rot_mat_inv).numpy()
            # results["extrinsics"][view] = (torch.tensor(results["extrinsics"][view]).float() @ rot_mat_inv).numpy()

        return