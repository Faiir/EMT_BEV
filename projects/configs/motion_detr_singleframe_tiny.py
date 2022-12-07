_base_ = [
    "../../configs/_base_/datasets/nus-3d.py",
    # "../../configs/_base_/schedules/cyclic_20e.py",
    "../../configs/_base_/schedules/cosine.py",
    "../../configs/_base_/default_runtime.py",
]

# 23351MB for single-GPU training
find_unused_parameters = False
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
sync_bn = True

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

# Image-view augmentation
data_aug_conf = {
    "resize_lim": (0.38, 0.55),
    "final_dim": (256, 704),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
    "bot_pct_lim": (0.0, 0.22),
    "crop_h": (0.0, 0.0),
    "cams": [
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
    ],
    "Ncams": 6,
}

bev_aug_params = {
    "rot_range": [-0.3925, 0.3925],
    "scale_range": [0.95, 1.05],
    "trans_std": [0, 0, 0],
    "hflip": 0.5,
    "vflip": 0.5,
}

det_grid_conf = {
    "xbound": [-51.2, 51.2, 0.8],
    "ybound": [-51.2, 51.2, 0.8],
    "zbound": [-10.0, 10.0, 20.0],
    "dbound": [1.0, 60.0, 1.0],
}

motion_grid_conf = {
    "xbound": [-50.0, 50.0, 0.5],
    "ybound": [-50.0, 50.0, 0.5],
    "zbound": [-10.0, 10.0, 20.0],
    "dbound": [1.0, 60.0, 1.0],
}

map_grid_conf = {
    "xbound": [-30.0, 30.0, 0.15],
    "ybound": [-15.0, 15.0, 0.15],
    "zbound": [-10.0, 10.0, 20.0],
    "dbound": [1.0, 60.0, 1.0],
}

grid_conf = det_grid_conf

receptive_field = 1
future_frames = 0
future_discount = 1.0

hidden_dim = 128
num_queries = 300
num_feature_levels = 3

voxel_size = [0.1, 0.1, 0.2]
model = dict(
    type="BEVerse_Motion_DETR",
    img_backbone=dict(
        type="SwinTransformer",
        pretrained="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(
            2,
            3,
        ),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        use_abs_pos_embed=False,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="LN", requires_grad=True),
        pretrain_style="official",
        output_missing_index_as_none=False,
    ),
    img_neck=dict(
        type="FPN_LSS",
        in_channels=384 + 768,
        inverse=True,
    ),
    transformer=dict(
        type="TransformerLSS",
        grid_conf=grid_conf,
        input_dim=data_aug_conf["final_dim"],
        numC_input=512,
        numC_Trans=64,
    ),
    temporal_model=dict(
        type="TemporalIdentity",
        grid_conf=grid_conf,
    ),
    pts_bbox_head=dict(
        type="MultiTaskHead_Motion_DETR",
        n_future=future_frames,
        in_channels=64,
        out_channels_det=64,
        out_channels_map=256,
        hidden_dim=hidden_dim,
        nheads=8,
        enc_layers=3,
        dec_layers=3,
        dec_n_points=20,
        enc_n_points=20,
        dim_feedforward=hidden_dim,
        dropout_transformer=0.1, 
        activation="relu",
        num_feature_levels=num_feature_levels,
        num_queries=num_queries,
        backbone="resnet18",
        position_embedding="sine",
        num_pos_feats=128,
        temporal_queries_activated=False, 
        flow_warp=False,
        grid_conf=grid_conf,
        det_grid_conf=det_grid_conf,
        map_grid_conf=map_grid_conf,
        motion_grid_conf=motion_grid_conf,
        using_ego=True,
        bbox_coder=dict(
            type="NMSFreeCoder",
            pc_range=point_cloud_range,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=150,
            #score_threshold=0.1,
            #out_size_factor=8,
            voxel_size=voxel_size,
            num_classes=10,
        ),
        task_enable={
            "3dod": False,
            "map": False,
            "motion": True,
        },
        task_weights={
            "3dod": 1.0,
            "map": 10.0,
            "motion": 1.0,
        },
        bev_encode_block="Basic",
        cfg_3dod=dict(
            type="Motion_DETR_DET",
            in_channels=256,
            hidden_dim=hidden_dim,
            tasks=[
                dict(num_class=1, class_names=["car"]),
                dict(num_class=2, class_names=["truck", "construction_vehicle"]),
                dict(num_class=2, class_names=["bus", "trailer"]),
                dict(num_class=1, class_names=["barrier"]),
                dict(num_class=2, class_names=["motorcycle", "bicycle"]),
                dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
            ],
            common_heads=dict(
                reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
            ),
            #share_conv_channel=64,
            n_future=future_frames,
            #separate_head=dict(type="SeparateHead", init_bias=-2.19, final_kernel=3),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0),
            loss_bbox=dict(type='L1Loss', loss_weight=0.25),
            loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        ),
        cfg_map=dict(
            type="MapHead",
            task_dict={
                "semantic_seg": 4,
            },
            in_channels=256,
            class_weights=[1.0, 2.0, 2.0, 2.0],
            semantic_thresh=0.25,
        ),
        cfg_motion=dict(
            type="Motion_DETR_MOT",
            #in_channels=256,
            hidden_dim=hidden_dim,
            num_feature_levels=num_feature_levels,
            mask_stride=4,
            nheads=8,
            num_queries=num_queries,
            grid_conf=motion_grid_conf,
            #class_weights=[1.0, 2.0],
            receptive_field=receptive_field,
            n_future=future_frames,
            future_discount=future_discount,

        ),
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                # Fake cost. This is just to make it compatible with DETR head.
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=point_cloud_range)
        )
    ),
    # test_cfg=dict(
    #     pts=dict(
    #         pc_range=point_cloud_range[:2],
    #         post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    #         max_per_img=500,
    #         max_pool_nms=False,
    #         min_radius=[4, 12, 10, 1, 0.85, 0.175],
    #         score_threshold=0.1,
    #         #out_size_factor=8,
    #         voxel_size=voxel_size[:2],
    #         out_size_factor=4,

    #     )
    # ),
)

dataset_type = "MTLEgoNuScenesDataset"
data_root = "/home/niklas/ETM_BEV/BEVerse/data/nuscenes/"
data_info_path = "/home/niklas/ETM_BEV/BEVerse/data/nuscenes_infos/"

train_pipeline = [
    # load image and apply image-view augmentation
    dict(
        type="LoadMultiViewImageFromFiles_MTL",
        using_ego=True,
        temporal_consist=True,
        is_train=True,
        data_aug_conf=data_aug_conf,
    ),
    # load 3D bounding boxes & bev-semantic-maps
    dict(
        type="LoadAnnotations3D_MTL",
        with_bbox_3d=True,
        with_label_3d=True,
        with_instance_tokens=True,
    ),
    # bev-augmentations
    dict(
        type="MTLGlobalRotScaleTrans",
        rot_range=bev_aug_params["rot_range"],
        scale_ratio_range=bev_aug_params["scale_range"],
        translation_std=bev_aug_params["trans_std"],
        update_img2lidar=True,
    ),
    dict(
        type="MTLRandomFlip3D",
        sync_2d=False,
        flip_ratio_bev_horizontal=bev_aug_params["hflip"],
        flip_ratio_bev_vertical=bev_aug_params["vflip"],
        update_img2lidar=True,
    ),
    # convert map labels
    dict(type="RasterizeMapVectors", map_grid_conf=map_grid_conf),
    # convert motion labels
    dict(type="ConvertMotionLabels", grid_conf=motion_grid_conf, only_vehicle=True),
    # filter objects
    dict(type="ObjectValidFilter"),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    # bundle & collect
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type="Collect3D",
        keys=[
            "img_inputs",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "semantic_indices",
            "semantic_map",
            "future_egomotions",
            "aug_transform",
            "img_is_valid",
            "motion_segmentation",
            "motion_instance",
            "instance_centerness",
            "instance_offset",
            "instance_flow",
            "has_invalid_frame",
        ],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "lidar2img",
            "depth2img",
            "cam2img",
            "pad_shape",
            "lidar2ego_rots",
            "lidar2ego_trans",
            "scale_factor",
            "flip",
            "pcd_horizontal_flip",
            "pcd_vertical_flip",
            "box_mode_3d",
            "box_type_3d",
            "img_norm_cfg",
            "pcd_trans",
            "sample_idx",
            "pcd_scale_factor",
            "pcd_rotation",
            "pts_filename",
            "transformation_3d_flow",
            "img_info",
        ),
    ),
]
test_pipeline = [
    dict(
        type="LoadMultiViewImageFromFiles_MTL",
        using_ego=True,
        data_aug_conf=data_aug_conf,
    ),
    dict(
        type="LoadAnnotations3D_MTL",
        with_bbox_3d=True,
        with_label_3d=True,
        with_instance_tokens=True,
    ),
    dict(type="RasterizeMapVectors", map_grid_conf=map_grid_conf),
    dict(type="ConvertMotionLabels", grid_conf=motion_grid_conf, only_vehicle=True),
    # filter objects
    # dict(type='TemporalObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='TemporalObjectNameFilter', classes=class_names),
    dict(type="ObjectValidFilter"),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="DefaultFormatBundle3D", class_names=class_names, with_label=False
            ),
            dict(
                type="Collect3D",
                keys=[
                    "img_inputs",
                    "semantic_indices",
                    "semantic_map",
                    "future_egomotions",
                    "gt_bboxes_3d",
                    "gt_labels_3d",
                    "motion_segmentation",
                    "motion_instance",
                    "instance_centerness",
                    "instance_offset",
                    "instance_flow",
                    "has_invalid_frame",
                    "img_is_valid",
                ],
                meta_keys=(
                    "filename",
                    "ori_shape",
                    "img_shape",
                    "lidar2img",
                    "depth2img",
                    "cam2img",
                    "pad_shape",
                    "scale_factor",
                    "flip",
                    "pcd_horizontal_flip",
                    "pcd_vertical_flip",
                    "box_mode_3d",
                    "box_type_3d",
                    "img_norm_cfg",
                    "pcd_trans",
                    "sample_idx",
                    "pcd_scale_factor",
                    "pcd_rotation",
                    "pts_filename",
                    "transformation_3d_flow",
                    "img_info",
                    "lidar2ego_rots",
                    "lidar2ego_trans",
                ),
            ),
        ],
    ),
]

input_modality = dict(
    use_camera=True,
    use_lidar=False,
    use_radar=False,
    use_map=False,
    use_external=False,
    prototype="lift-splat-shoot",
)

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        type="CBGSDataset",
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_info_path + "nuscenes_infos_train.pkl",
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            use_valid_flag=True,
            receptive_field=receptive_field,
            future_frames=future_frames,
            grid_conf=grid_conf,
            map_grid_conf=map_grid_conf,
            modality=input_modality,
            box_type_3d="LiDAR",
        ),
    ),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        receptive_field=receptive_field,
        future_frames=future_frames,
        grid_conf=grid_conf,
        map_grid_conf=map_grid_conf,
        ann_file=data_info_path + "nuscenes_infos_val.pkl",
        modality=input_modality,
    ),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        receptive_field=receptive_field,
        future_frames=future_frames,
        grid_conf=grid_conf,
        map_grid_conf=map_grid_conf,
        ann_file=data_info_path + "nuscenes_infos_val.pkl",
        modality=input_modality,
    ),
)

optimizer = dict(type="AdamW", betas=(0.95, 0.99),
                 lr=2e-4, weight_decay=0.0001)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)

evaluation = dict(interval=999, pipeline=test_pipeline)
