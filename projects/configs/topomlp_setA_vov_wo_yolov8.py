custom_imports = dict(imports=['projects.topomlp'])

method_para = dict(n_control=4) # number of control points

_dim_ = 256
point_cloud_range = [-51.2, -25.6, -8, 51.2, 25.6, 4]
voxel_size = [0.2, 0.2, 8]
SyncBN = True

model = dict(
    type='TopoMLP',
    use_grid_mask=True,
    img_backbone=dict(
        type='VoVNetCP',  ###use checkpoint to save memory
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=('stage2', 'stage3', 'stage4', 'stage5',)),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 768, 1024],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True,
    ),
    lc_head=dict(
            type='LaneHead',
            num_classes=1,
            in_channels=_dim_,
            num_lane=1800,   # 300+1500
            num_lanes_one2one=300,
            k_one2many=5,
            lambda_one2many=2.0,
            num_reg_dim=method_para['n_control'] * 3,
            transformer=dict(
                type='PETRTransformer',
                decoder=dict(
                    type='PETRTransformerDecoder',
                    return_intermediate=True,
                    num_layers=6,
                    transformerlayers=dict(
                        type='PETRTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1),
                            dict(
                                type='PETRMultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1),
                            ],
                        feedforward_channels=2048,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                         'ffn', 'norm')),
                )),
            positional_encoding=dict(
                type='SinePositionalEncoding3D', num_feats=128, normalize=True),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.5),
            loss_bbox=dict(type='L1Loss', loss_weight=0.02),
            loss_iou=dict(type='GIoULoss', loss_weight=0.0), # dummy
            train_cfg=dict(
                assigner=dict(
                    type='LaneHungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=1.5),
                    reg_cost=dict(type='LaneL1Cost', weight=0.02),
                    iou_cost=dict(type='IoUCost', weight=0.0))), # dummy
            bev_range=point_cloud_range,
            LID=True,
            with_position=True,
            with_multiview=True,
            position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            normedlinear=False,
            ),
    te_head=dict(
        type='TrafficHead',
        num_query=100,
        num_classes=13,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='DeformableDetrTransformer',
            num_feature_levels=4,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=_dim_),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=_dim_,
                        feedforward_channels=512,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='CustomDetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=_dim_)
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=_dim_,
                        feedforward_channels=512,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=2.5),
        loss_iou=dict(type='GIoULoss', loss_weight=1.0),
        test_cfg=dict(max_per_img=50),
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=1.0),
                reg_cost=dict(type='BBoxL1Cost', weight=2.5, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0))),
    ),
    lclc_head=dict(
        type='TopoLLHead',
        in_channels_o1=_dim_,
        in_channels_o2=_dim_,
        shared_param=False,
        loss_rel=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=5),
        loss_ll_l1_weight=0.1,
        add_lane_pred=True,
        lane_pred_dimension=method_para['n_control'] * 3,
        is_detach=True),
    lcte_head=dict(
        type='TopoLTHead',
        in_channels_o1=_dim_,
        in_channels_o2=_dim_,
        shared_param=False,
        loss_rel=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=5),
        add_pos=True,
        pos_dimension=9,
        is_detach=True),)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='ResizeFrontView'),
    dict(type='ResizeMultiview3D', img_scale=(1024, 775), keep_ratio=False),
    # dict(type='GlobalRotScaleTransImage',
    #      rot_range=[-0.3925, 0.3925],
    #      translation_std=[0, 0, 0],
    #      scale_ratio_range=[0.95, 1.05],
    #      training=True
    #      ),
    dict(type='CustomPadMultiViewImage', size_divisor=32),
    dict(type='CustomParameterizeLane', method='bezier_Endpointfixed', method_para=method_para),
    dict(type='CustomDefaultFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'img',
            'gt_lc', 'gt_lc_labels',
            'gt_te', 'gt_te_labels',
            'gt_topology_lclc', 'gt_topology_lcte',
        ],
        meta_keys=[
            'scene_token', 'sample_idx', 'img_paths',
            'img_shape', 'scale_factor', 'pad_shape',
            'lidar2img', 'can_bus', 'te_yolov8',
        ],
    )
]

test_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='ResizeFrontView'),
    dict(type='ResizeMultiview3D', img_scale=(1024, 775), keep_ratio=False),
    dict(type='CustomPadMultiViewImage', size_divisor=32),
    dict(type='CustomDefaultFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'img',
        ],
        meta_keys=[
            'scene_token', 'sample_idx', 'img_paths',
            'img_shape', 'scale_factor', 'pad_shape',
            'lidar2img', 'can_bus', 'te_yolov8',
        ],
    )
]

dataset_type = 'OpenLaneV2SubsetADataset'
data_root = './data'
meta_root = './data'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        meta_root=meta_root,
        collection='data_dict_subset_A_train',
        pipeline=train_pipeline,
        test_mode=False,
        yolov8_file=None),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        meta_root=meta_root,
        collection='data_dict_subset_A_val',
        pipeline=test_pipeline,
        test_mode=True,
        yolov8_file=None),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        meta_root=meta_root,
        collection='data_dict_subset_A_val',
        pipeline=test_pipeline,
        test_mode=True,
        yolov8_file=None),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
            custom_keys={
                'img_backbone': dict(lr_mult=0.2),
            }),
    weight_decay=1e-2)
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic', grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(interval=24, pipeline=test_pipeline, visualization_num=300)

checkpoint_config = dict(interval=1, max_keep_ckpts=3)

# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = None
load_from = '/data/wudongming/mmdetection3d/ckpts/fcos3d_vovnet_imgbackbone-remapped.pth'
resume_from = None
workflow = [('train', 1)]
