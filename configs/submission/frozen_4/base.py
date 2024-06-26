_base_ = [
    "../../../mmdetection3d/configs/_base_/datasets/nus-3d.py",
    "../../mlflow_runtime.py",
]

custom_imports = dict(
    imports=['spatial_detr.models.utils',
             'detr3d.models'], allow_failed_imports=False)


_samples_per_gpu = 1
_workers_per_gpu = 1
frozen = 4
# _img_scale = 0.2

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(mean=[103.530, 116.280, 123.675],
                    std=[1.0, 1.0, 1.0], to_rgb=False)
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

input_modality = dict(
    use_lidar=True, use_camera=True, use_radar=False, use_map=False, use_external=False
)

model_cls = "Detr3D"
model = dict(
    type=model_cls,
    use_grid_mask=True,
    img_backbone=dict(
        type="ResNet",
        depth=101,
        num_stages=4,
        out_indices=(3,),  # (0,1,2,3) # (for fpn)
        frozen_stages=frozen,  # freeze to reduce memory footprint
        norm_cfg=dict(type="BN2d", requires_grad=False),
        norm_eval=True,
        style="caffe",
        dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
    ),
    img_neck=dict(
        type='FPN',
        # adapt according to desired multi-scale lvls
        in_channels=[2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=1,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type="Detr3DHead",
        num_query=900,
        num_classes=10,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type="SpatialDETRTransformer",
            pos_encoding=dict(
                type="FixedGeometricEncoding",
                # encodings are in  sensor coordinates of each cam
                apply_global_rot=False,
            ),
            decoder=dict(
                type="MixedScaleSpatialDETRTransformerDecoder",
                num_layers=6,
                shared=False,
                switch_scale_layer_num=-1,
                return_intermediate=True,
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type="MultiheadAttention",
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                        ),
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
        ),
        bbox_coder=dict(
            type="NMSFreeCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10,
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        loss_iou=dict(type="GIoULoss", loss_weight=0.0),
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type="HungarianAssigner3D",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                # Fake cost. This is just to make it compatible with DETR head.
                iou_cost=dict(type="IoUCost", weight=0.0),
                pc_range=point_cloud_range,
            ),
        )
    ),
)

dataset_type = "CustomNuScenesDataset"
data_root = "data/nuscenes/"

file_client_args = dict(backend="disk")

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + "nuscenes_dbinfos_train.pkl",
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5,
        ),
    ),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2,
    ),
    points_loader=dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
    ),
)

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
    ),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type="Collect3D",
        keys=["gt_bboxes_3d", "gt_labels_3d", "img"],
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
            "cam_intrinsic",
            "lidar2cam",
        ),
    ),
]
test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
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
                keys=["img", "points"],
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
                    "cam_intrinsic",
                    "lidar2cam",
                ),
            ),
        ],
    ),
]


data = dict(
    samples_per_gpu=_samples_per_gpu,
    workers_per_gpu=_workers_per_gpu,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "nuscenes_infos_train.pkl",
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d="LiDAR",
    ),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
    ),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
    ),
)

total_epochs = 24
evaluation = dict(interval=2, pipeline=test_pipeline)

runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
load_from = "work_dirs/pretrained/fcos3d.pth"

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook"),
        dict(
            type="MlflowLoggerHook",
            tags={
                "total_epochs": total_epochs,
                "samples_per_gpu": _samples_per_gpu,
                "workers_per_gpu": _workers_per_gpu,
                "point_cloud_range": ", ".join(str(e) for e in point_cloud_range),
                "voxel_size": ", ".join(str(e) for e in voxel_size),
                "class_names": ", ".join(str(e) for e in class_names),
            },
        ),
    ],
)