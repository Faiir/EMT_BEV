# inspired by https://github.com/pytorch/kineto/blob/main/tb_plugin/examples/resnet50_profiler_api.py
import os

import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
from custome_logger import setup_custom_logger
import logging

logger = setup_custom_logger()
logger.debug("Profiler")


from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.cnn import fuse_conv_bn
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from mmcv.parallel import MMDataParallel
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor

pt_profiler = True


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

    return cfg


def import_modules_load_config(cfg_file="beverse_tiny.py", samples_per_gpu=1):
    cfg_path = r"/content/EMT_BEV/projects/configs"
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


def perform_10_steps(cfg, p):
    samples_per_gpu = 1

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    wrap_fp16_model(model)
    model.cuda()
    model = MMDataParallel(model, device_ids=[0])
    iter_loader = iter(data_loader)
    samples = []
    for i in range(10):
        samples.append(next(iter_loader))

    for sample in samples:

        motion_distribution_targets = {
            # for motion prediction
            "motion_segmentation": sample["motion_segmentation"][0],
            "motion_instance": sample["motion_instance"][0],
            "instance_centerness": sample["instance_centerness"][0],
            "instance_offset": sample["instance_offset"][0],
            "instance_flow": sample["instance_flow"][0],
            "future_egomotion": sample["future_egomotions"][0],
        }

        with torch.no_grad():
            result = model(
                return_loss=False,
                rescale=True,
                img_metas=sample["img_metas"],
                img_inputs=sample["img_inputs"],
                future_egomotions=sample["future_egomotions"],
                motion_targets=motion_distribution_targets,
                img_is_valid=sample["img_is_valid"][0],
            )
        if pt_profiler:
            p.step()


def main() -> None:
    device = torch.device("cuda:0")
    print("IterativeFlow")
    logger = logging.getLogger("timelogger")
    # Define different settings to test

    # img backbones

    resize_lims = [
        (0.3, 0.45),  # fiery
        (0.38, 0.55),  # desTINY
        (0.82, 0.99),  # small
        (1, 1),  # BEVDEt
    ]

    final_dims = [(224, 480), (256, 704), (512, 1408), (900, 1600)]

    backbones = [
        "beverse_tiny.py",
        "beverse_tiny.py",
        "beverse_small.py",
        "beverse_small.py",
    ]

    # future frames -> tiny settings
    future_frames_list = [4, 4, 4, 4, 5, 7, 10]
    receptive_field_list = [
        3,
        5,
        8,
        13,
        4,
        6,
        9,
    ]

    # grid_size = (
    #     point_cloud_range[3:] -  # type: ignore
    #     point_cloud_range[:3]) / voxel_size  # type: ignore

    point_cloud_range_base = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    point_cloud_range_extended_fustrum = [-71.2, -71.2, -5.0, 71.2, 71.2, 3.0]
    det_grid_confs = {
        "xbound": [
            [-51.2, 51.2, 0.8],  # lower_bound, upper_bound, interval
            [-51.2, 51.2, 0.4],
            [-51.2, 51.2, 0.2],
            [-51.2, 51.2, 0.1],
            [-26.2, 26.2, 0.8],
            [-26.2, 26.2, 0.4],
        ],
        "ybound": [
            [-51.2, 51.2, 0.8],
            [-51.2, 51.2, 0.4],
            [-51.2, 51.2, 0.2],
            [-51.2, 51.2, 0.1],
            [-71.2, 71.2, 0.8],
            [-71.2, 71.2, 0.4],
        ],
        "zbound": [-10.0, 10.0, 20.0],
        "dbound": [
            [1.0, 60.0, 1.0],
            [1.0, 60.0, 1.0],
            [1.0, 60.0, 0.5],
            [1.0, 60.0, 0.5],
            [1.0, 70.0, 1.0],
            [1.0, 70.0, 5.0],
        ],  # [(lower_bound, upper_bound, interval).]
    }

    motion_grid_confs = {
        "xbound": [
            [-50.0, 50.0, 0.5],
            [-50.0, 50.0, 0.25],
            [-50.0, 50.0, 0.125],
            [-50.0, 50.0, 0.075],
            [-25.0, 25.0, 0.5],
            [-25.0, 25.0, 0.25],
        ],
        "ybound": [
            [-50.0, 50.0, 0.5],
            [-50.0, 50.0, 0.25],
            [-50.0, 50.0, 0.125],
            [-50.0, 50.0, 0.075],
            [-70.0, 70.0, 0.5],
            [-70.0, 70.0, 0.25],
        ],
        "zbound": [-10.0, 10.0, 20.0],
        "dbound": [
            [1.0, 60.0, 1.0],
            [1.0, 60.0, 1.0],
            [1.0, 60.0, 0.5],
            [1.0, 60.0, 0.5],
            [1.0, 70.0, 1.0],
            [1.0, 70.0, 5.0],
        ],
    }

    map_grid_confs = {
        "xbound": [-30.0, 30.0, 0.15],
        "ybound": [-15.0, 15.0, 0.15],
        "zbound": [-10.0, 10.0, 20.0],
        "dbound": [
            [1.0, 60.0, 1.0],
            [1.0, 60.0, 1.0],
            [1.0, 60.0, 0.5],
            [1.0, 60.0, 0.5],
            [1.0, 70.0, 1.0],
            [1.0, 70.0, 5.0],
        ],
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
    # grid_confs = (det_grid_conf, motion_grid_conf, map_grid_conf)

    # First test settings differently and then select interesting combinations based on findings
    for c, (backbone, resize_lim, final_dim) in enumerate(
        zip(backbones, resize_lims, final_dims)
    ):
        cfg = import_modules_load_config(cfg_file=backbone)
        cfg = update_cfg(resize_lim=resize_lim, final_dim=final_dim)
        cfg["data_aug_conf"]["resize_lim"] = resize_lim
        cfg["data_aug_conf"]["final_dim"] = final_dims

        if pt_profiler:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    f"/content/drive/MyDrive/logs_thesis/logs_profiler/size_logs_{c}",
                    worker_name="worker0",
                ),
                record_shapes=True,
                profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
                with_stack=True,
            ) as p:
                perform_10_steps(cfg, p)
        else:
            perform_10_steps(cfg, None)

        logger.debug(
            "******" * 6
            + " resize_lim "
            + str(resize_lim)
            + " final_dim: "
            + str(final_dim)
            + "******" * 6
        )
    logger.debug("*******" * 12)

    for c, (future_frames, receptive_field) in enumerate(
        zip(future_frames_list, receptive_field_list)
    ):
        cfg = import_modules_load_config()
        cfg = update_cfg(future_frames=future_frames, receptive_field=receptive_field)
        cfg["future_frames"] = future_frames
        cfg["receptive_field"] = receptive_field

        if pt_profiler:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    f"/content/drive/MyDrive/logs_thesis/logs_profiler/future_frames_{c}",
                    worker_name="worker0",
                ),
                record_shapes=True,
                profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
                with_stack=True,
            ) as p:
                perform_10_steps(cfg, p)
        else:
            perform_10_steps(cfg, None)

        logger.debug(
            "******" * 6
            + " future_frames "
            + str(future_frames)
            + " receptive_field: "
            + str(receptive_field)
            + "******" * 6
        )
    logger.debug("*******" * 12)

    for i, d in enumerate(map_grid_confs["dbound"]):
        det_grid_conf["xbound"] = det_grid_confs["xbound"][i]
        det_grid_conf["ybound"] = det_grid_confs["ybound"][i]
        det_grid_conf["zbound"] = det_grid_confs["zbound"]
        det_grid_conf["dbound"] = det_grid_confs["dbound"][i]

        motion_grid_conf["xbound"] = motion_grid_confs["xbound"][i]
        motion_grid_conf["ybound"] = motion_grid_confs["ybound"][i]
        motion_grid_conf["zbound"] = motion_grid_confs["zbound"]
        motion_grid_conf["dbound"] = motion_grid_confs["dbound"][i]

        map_grid_conf["xbound"] = det_grid_confs["xbound"]
        map_grid_conf["ybound"] = det_grid_confs["ybound"]
        map_grid_conf["zbound"] = det_grid_confs["zbound"]
        map_grid_conf["dbound"] = det_grid_confs["dbound"][i]

        cfg = import_modules_load_config()
        cfg = update_cfg(
            grid_conf=det_grid_conf,
            det_grid_conf=det_grid_conf,
            motion_grid_conf=motion_grid_conf,
            map_grid_conf=map_grid_conf,
        )
        cfg["det_grid_conf"] = det_grid_conf
        cfg["motion_grid_conf"] = motion_grid_conf
        cfg["map_grid_conf"] = map_grid_conf
        cfg["grid_conf"] = det_grid_conf
        if i <= 4:
            cfg["point_cloud_range"] = point_cloud_range_base
        else:
            cfg["point_cloud_range"] = point_cloud_range_extended_fustrum

        if pt_profiler:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    f"/content/drive/MyDrive/logs_thesis/logs_profiler/grid_config_{i}",
                    worker_name="worker0",
                ),
                record_shapes=True,
                profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
                with_stack=True,
            ) as p:
                perform_10_steps(cfg, p)
        else:
            perform_10_steps(cfg, None)

        logger.debug(
            "******" * 6 + " det_grid_conf " + str(det_grid_conf) + "******" * 6
        )
    logger.debug("*******" * 12)


if __name__ == "__main__":
    main()
