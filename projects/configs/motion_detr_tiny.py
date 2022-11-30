_base_ = ["./motion_detr_singleframe_tiny.py"]

receptive_field = 3
future_frames = 4
future_discount = 0.95

model = dict(
    temporal_model=dict(
        type="Temporal3DConvModel",
        receptive_field=receptive_field,
        input_egopose=True,
        in_channels=64,
        input_shape=(128, 128),
        with_skip_connect=True,
    ),
    pts_bbox_head=dict(
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
        cfg_motion=dict(
            type="Motion_DETR_MOT",
            # task_dict={
            #     "segmentation": 2,
            #     "instance_center": 1,
            #     "instance_offset": 2,
            #     # 'instance_flow': 2,
            # },
            #in_channels=256,
            hidden_dim=256,
            nheads=8,
            num_queries=300,
            #class_weights=[1.0, 2.0],
            receptive_field=receptive_field,
            n_future=future_frames,
            future_discount=future_discount,
            aux_loss=True, 
            matcher_config={
                "type": "HungarianMatcherIFC",
                "cost_class": 1,
                "cost_dice": 3.0,
                "num_classes": 100
            },
            criterion_config={
                "num_classes": 100,
                "weight_dict": {"loss_ce": 1, "loss_mask": 3.0,
                                "loss_dice": 3.0},
                "eos_coef": 0.1,
                "losses": ["labels", "masks", "cardinality"],
            },
            dec_layers=6,
    ),),
    train_cfg=dict(
        pts=dict(
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    ),
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        dataset=dict(
            receptive_field=receptive_field,
            future_frames=future_frames,
        ),
    ),
    val=dict(
        receptive_field=receptive_field,
        future_frames=future_frames,
    ),
    test=dict(
        receptive_field=receptive_field,
        future_frames=future_frames,
    ),
)
