# benchmark_pytorch.py from https://leimao.github.io/blog/PyTorch-Benchmark/
import os
from timeit import default_timer as timer
import torch
import torch.nn as nn

# import torchvision
import torch.utils.benchmark as benchmark

from mmcv import Config

# from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

# from mmdet3d.models import build_detector
# from tools.misc.fuse_conv_bn import fuse_module


from mmdet3d.models import build_model


torch.backends.cudnn.benchmark = True


@torch.no_grad()
def measure_time_host(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_repeats: int = 100,
    num_warmups: int = 10,
    synchronize: bool = True,
    continuous_measure: bool = True,
) -> float:

    for _ in range(num_warmups):
        _ = model.forward(input_tensor)
    torch.cuda.synchronize()

    elapsed_time_ms = 0

    if continuous_measure:
        start = timer()
        for _ in range(num_repeats):
            _ = model.forward(input_tensor)
        if synchronize:
            torch.cuda.synchronize()
        end = timer()
        elapsed_time_ms = (end - start) * 1000

    else:
        for _ in range(num_repeats):
            start = timer()
            _ = model.forward(input_tensor)
            if synchronize:
                torch.cuda.synchronize()
            end = timer()
            elapsed_time_ms += (end - start) * 1000

    return elapsed_time_ms / num_repeats


@torch.no_grad()
def measure_time_device(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_repeats: int = 100,
    num_warmups: int = 10,
    synchronize: bool = True,
    continuous_measure: bool = True,
) -> float:

    for _ in range(num_warmups):
        _ = model.forward(input_tensor)
    torch.cuda.synchronize()

    elapsed_time_ms = 0

    if continuous_measure:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_repeats):
            _ = model.forward(input_tensor)
        end_event.record()
        if synchronize:
            # This has to be synchronized to compute the elapsed time.
            # Otherwise, there will be runtime error.
            torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)

    else:
        for _ in range(num_repeats):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            _ = model.forward(input_tensor)
            end_event.record()
            if synchronize:
                # This has to be synchronized to compute the elapsed time.
                # Otherwise, there will be runtime error.
                torch.cuda.synchronize()
            elapsed_time_ms += start_event.elapsed_time(end_event)

    return elapsed_time_ms / num_repeats


@torch.no_grad()
def run_inference(model: nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:

    return model.forward(input_tensor)


def main() -> None:

    num_warmups = 100
    num_repeats = 1000
    # Change to C x 1 x 3 x 1600 x 900
    # 704×256 Tiny
    # 1408×512
    input_shape = (6, 1600, 900, 3)

    device = torch.device("cuda:0")
    cfg = Config.fromfile(
        r"/home/niklas/ETM_BEV/BEVerse/projects/configs/beverse_tiny.py"
    )

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
                _module_dir = os.path.dirname(
                    r"/home/niklas/ETM_BEV/BEVerse/projects/configs/beverse_tiny.py"
                )
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
    # model = torchvision.models.resnet18(pretrained=False)
    # model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    wrap_fp16_model(model)
    # load_checkpoint(
    #     model,
    #     r"/home/niklas/ETM_BEV/BEVerse/checkpoints/beverse_tiny.pth",
    #     map_location="cpu",
    # )
    # model = fuse_module(model)
    model.cuda(device)
    model.eval()
    # model = nn.Conv2d(in_channels=input_shape[1], out_channels=256, kernel_size=(5, 5))

    # Input tensor
    input_tensor = torch.rand(input_shape, device=device)

    torch.cuda.synchronize()

    print("Latency Measurement Using CPU Timer...")
    for continuous_measure in [True]:
        for synchronize in [True]:
            try:
                latency_ms = measure_time_host(
                    model=model,
                    input_tensor=input_tensor,
                    num_repeats=num_repeats,
                    num_warmups=num_warmups,
                    synchronize=synchronize,
                    continuous_measure=continuous_measure,
                )
                print(
                    f"|"
                    f"Synchronization: {synchronize!s:5}| "
                    f"Continuous Measurement: {continuous_measure!s:5}| "
                    f"Latency: {latency_ms:.5f} ms| "
                )
            except Exception as e:
                print(
                    f"|"
                    f"Synchronization: {synchronize!s:5}| "
                    f"Continuous Measurement: {continuous_measure!s:5}| "
                    f"Latency: N/A     ms| "
                )
            torch.cuda.synchronize()

    print("Latency Measurement Using CUDA Timer...")
    for continuous_measure in [True, False]:
        for synchronize in [True, False]:
            try:
                latency_ms = measure_time_device(
                    model=model,
                    input_tensor=input_tensor,
                    num_repeats=num_repeats,
                    num_warmups=num_warmups,
                    synchronize=synchronize,
                    continuous_measure=continuous_measure,
                )
                print(
                    f"|"
                    f"Synchronization: {synchronize!s:5}| "
                    f"Continuous Measurement: {continuous_measure!s:5}| "
                    f"Latency: {latency_ms:.5f} ms| "
                )
            except Exception as e:
                print(
                    f"|"
                    f"Synchronization: {synchronize!s:5}| "
                    f"Continuous Measurement: {continuous_measure!s:5}| "
                    f"Latency: N/A     ms| "
                )
            torch.cuda.synchronize()

    print("Latency Measurement Using PyTorch Benchmark...")
    num_threads = 1
    timer = benchmark.Timer(
        stmt="run_inference(model, input_tensor)",
        setup="from __main__ import run_inference",
        globals={"model": model, "input_tensor": input_tensor},
        num_threads=num_threads,
        label="Latency Measurement",
        sub_label="torch.utils.benchmark.",
    )

    profile_result = timer.timeit(num_repeats)
    # https://pytorch.org/docs/stable/_modules/torch/utils/benchmark/utils/common.html#Measurement
    print(f"Latency: {profile_result.mean * 1000:.5f} ms")


if __name__ == "__main__":

    main()
