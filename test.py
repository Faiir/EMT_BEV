import os

import torch

from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.runner import load_checkpoint

from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor

from mmcv import Config


def update_cfg(
    cfg,
    n_future=3,
    receptive_field=3,
    resize_lim=(0.38, 0.55),
    final_dim=(256, 704),
    grid_conf={
        "xbound": [-51.2, 51.2, 0.8],
        "ybound": [-51.2, 51.2, 0.8],
        "zbound": [-10.0, 10.0, 20.0],
        "dbound": [1.0, 60.0, 1.0],
    },
    det_grid_conf={
        "xbound": [-51.2, 51.2, 0.8],
        "ybound": [-51.2, 51.2, 0.8],
        "zbound": [-10.0, 10.0, 20.0],
        "dbound": [1.0, 60.0, 1.0],
    },
    map_grid_conf={
        "xbound": [-30.0, 30.0, 0.15],
        "ybound": [-15.0, 15.0, 0.15],
        "zbound": [-10.0, 10.0, 20.0],
        "dbound": [1.0, 60.0, 1.0],
    },
    motion_grid_conf={
        "xbound": [-50.0, 50.0, 0.5],
        "ybound": [-50.0, 50.0, 0.5],
        "zbound": [-10.0, 10.0, 20.0],
        "dbound": [1.0, 60.0, 1.0],
    },
    t_input_shape=(128, 128),
    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
):

    cfg["det_grid_conf"] = det_grid_conf
    cfg["map_grid_conf"] = map_grid_conf
    cfg["motion_grid_conf"] = motion_grid_conf
    cfg["grid_conf"] = det_grid_conf

    cfg["model"]["temporal_model"]["input_shape"] = t_input_shape

    cfg["data"]["val"]["pipeline"][0]["data_aug_conf"]["resize_lim"] = resize_lim
    cfg["data"]["train"]["dataset"]["pipeline"][0]["data_aug_conf"][
        "resize_lim"
    ] = resize_lim
    cfg["data"]["val"]["pipeline"][0]["data_aug_conf"]["final_dim"] = final_dim
    cfg["data"]["train"]["dataset"]["pipeline"][0]["data_aug_conf"][
        "final_dim"
    ] = final_dim

    cfg["data"]["test"]["pipeline"][0]["data_aug_conf"]["resize_lim"] = resize_lim
    cfg["data"]["test"]["pipeline"][0]["data_aug_conf"]["final_dim"] = final_dim

    cfg["model"]["pts_bbox_head"]["cfg_motion"][
        "grid_conf"
    ] = motion_grid_conf  # motion_grid
    cfg["model"]["temporal_model"]["grid_conf"] = grid_conf
    cfg["model"]["transformer"]["grid_conf"] = grid_conf
    cfg["model"]["pts_bbox_head"]["grid_conf"] = grid_conf
    cfg["data"]["train"]["dataset"]["grid_conf"] = motion_grid_conf
    cfg["data"]["val"]["pipeline"][3]["grid_conf"] = motion_grid_conf
    cfg["data"]["test"]["pipeline"][3]["grid_conf"] = motion_grid_conf
    cfg["data"]["val"]["grid_conf"] = grid_conf
    cfg["data"]["test"]["grid_conf"] = grid_conf

    cfg["model"]["pts_bbox_head"]["det_grid_conf"] = det_grid_conf

    cfg["model"]["pts_bbox_head"]["map_grid_conf"] = map_grid_conf
    cfg["data"]["train"]["dataset"]["map_grid_conf"] = map_grid_conf
    cfg["data"]["test"]["pipeline"][2]["map_grid_conf"] = map_grid_conf
    cfg["data"]["val"]["pipeline"][2]["map_grid_conf"] = map_grid_conf
    cfg["data"]["test"]["map_grid_conf"] = map_grid_conf
    cfg["data"]["val"]["map_grid_conf"] = map_grid_conf

    cfg["model"]["pts_bbox_head"]["motion_grid_conf"] = motion_grid_conf

    cfg["data"]["test"]["pipeline"][5]["point_cloud_range"] = point_cloud_range
    cfg["data"]["train"]["pipeline"][5][
        "point_cloud_range"
    ] = point_cloud_range  # point_cloud_range=None
    cfg["data"]["train"]["pipeline"][6][
        "point_cloud_range"
    ] = point_cloud_range  #'point_cloud_range =None
    cfg["data"]["val"]["pipeline"][5]["point_cloud_range"] = point_cloud_range

    cfg["model"]["pts_bbox_head"]["cfg_motion"]["receptive_field"] = receptive_field
    cfg["data"]["train"]["dataset"]["receptive_field"] = receptive_field
    cfg["model"]["temporal_model"]["receptive_field"] = receptive_field
    cfg["data"]["test"]["receptive_field"] = receptive_field
    cfg["data"]["val"]["receptive_field"] = receptive_field

    cfg["data"]["val"]["future_frames"] = n_future
    cfg["model"]["pts_bbox_head"]["cfg_motion"]["n_future"] = n_future
    cfg["data"]["test"]["future_frames"] = n_future
    cfg["data"]["train"]["dataset"]["future_frames"] = n_future

    cfg["data"]["train"]["dataset"]["pipeline"][4]["map_grid_conf"] = map_grid_conf
    cfg["data"]["train"]["dataset"]["pipeline"][5]["grid_conf"] = motion_grid_conf
    cfg["data"]["train"]["dataset"]["pipeline"][7][
        "point_cloud_range"
    ] = point_cloud_range

    return cfg


def import_modules_load_config(cfg_file="beverse_tiny.py", samples_per_gpu=1):
    cfg_path = r"/home/niklas/ETM_BEV/BEVerse/projects/configs"
    cfg_path = os.path.join(cfg_path, cfg_file)

    cfg = Config.fromfile(cfg_path)

    # if args.cfg_options is not None:
    #     cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, "plugin"):
        if cfg.plugin:
            import importlib

            if hasattr(cfg, "plugin_dir"):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = cfg_path
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    return cfg


torch.backends.cudnn.benchmark = True


cfg = import_modules_load_config(
    cfg_file="/home/niklas/ETM_BEV/BEVerse/projects/configs/beverse_tiny_exp.py"
)


"model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
