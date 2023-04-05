# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
import mmcv
import torch
from mmcv.image import tensor2imgs
from os import path as osp
import pdb
import time

import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

# define semantic metrics
from ..metrics import IntersectionOverUnion, PanopticMetric
from ..visualize import Visualizer

from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table
from ..models.utils.vis_utils import generate_instance_colours, plot_instance_map

INSTANCE_COLOURS = np.asarray([
    [0, 0, 0],
    [255, 179, 0],
    [128, 62, 117],
    [255, 104, 0],
    [166, 189, 215],
    [193, 0, 32],
    [206, 162, 98],
    [129, 112, 102],
    [0, 125, 52],
    [246, 118, 142],
    [0, 83, 138],
    [255, 122, 92],
    [83, 55, 122],
    [255, 142, 0],
    [179, 40, 81],
    [244, 200, 0],
    [127, 24, 13],
    [147, 170, 0],
    [89, 51, 21],
    [241, 58, 19],
    [35, 44, 22],
    [112, 224, 255],
    [70, 184, 160],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [0, 255, 235],
    [255, 0, 235],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 255, 204],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [255, 214, 0],
    [25, 194, 194],
    [92, 0, 255],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
])



def single_gpu_test_motion_detr(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """

    model.eval()
    dataset = data_loader.dataset
    # whether for test submission
    test_mode = dataset.test_submission

    # bev coordinate system, LiDAR or ego
    coordinate_system = dataset.coordinate_system

    prog_bar = mmcv.ProgressBar(len(dataset))


    # logging interval
    logging_interval = 50

    # whether each task is enabled
    task_enable = model.module.pts_bbox_head.task_enable
    motion_enable = task_enable.get('motion', True)
    det_results = []


    # evaluate motion in (short, long) ranges
    EVALUATION_RANGES = {'30x30': (70, 130), '100x100': (0, 200)}
    num_motion_class = 2

    #motion_panoptic_metrics = {}
    motion_iou_metrics = {}
    for key in EVALUATION_RANGES.keys():
        # motion_panoptic_metrics[key] = PanopticMetric(
        #     n_classes=num_motion_class, temporally_consistent=True).cuda()
        motion_iou_metrics[key] = IntersectionOverUnion(
            num_motion_class).cuda()

    motion_eval_count = 0

    latencies = []
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            if test_mode:
                motion_distribution_targets = None
            else:
                motion_distribution_targets = {
                    # for motion prediction
                    'motion_segmentation': data['motion_segmentation'][0],
                    'motion_instance': data['motion_instance'][0],
                    'instance_centerness': data['instance_centerness'][0],
                    'instance_offset': data['instance_offset'][0],
                    'instance_flow': data['instance_flow'][0],
                    'future_egomotion': data['future_egomotions'][0],
                }

            result = model(
                return_loss=False,
                rescale=True,
                img_metas=data['img_metas'],
                img_inputs=data['img_inputs'],
                future_egomotions=data['future_egomotions'],
                motion_targets=motion_distribution_targets,
                img_is_valid=data['img_is_valid'][0],
            )

        # motion prediction results
        if motion_enable:
            motion_segmentation =   result['mask_pred_reduced'][0] #! only batchsize 0 accepted due to frame checking
            #motion_instance = result['motion_instance']
            has_invalid_frame = data['has_invalid_frame'][0]
            # valid future frames < n_future_frame, skip the evaluation
            if not has_invalid_frame.item():
                motion_eval_count += 1
                if not test_mode:
                    # generate targets
                    motion_targets = {
                        'motion_segmentation': data['motion_segmentation'][0],
                        'motion_instance': data['motion_instance'][0],
                        'instance_centerness': data['instance_centerness'][0],
                        'instance_offset': data['instance_offset'][0],
                        'instance_flow': data['instance_flow'][0],
                        'future_egomotion': data['future_egomotions'][0],
                    }
                    motion_labels, _ = model.module.pts_bbox_head.task_decoders['motion'].prepare_future_labels(
                        motion_targets)
                    motion_labels_seg = motion_labels[0]["match_masks"].transpose(1,0).sum(1).unsqueeze(0).unsqueeze(2)
                    #motion_labels_seg = motion_labels[0]["segmentation"].unsqueeze(0).unsqueeze(2)
                    for key, grid in EVALUATION_RANGES.items():
                        limits = slice(grid[0], grid[1])
                        # motion_panoptic_metrics[key](motion_instance[..., limits, limits].contiguous(
                        # ), motion_labels['instance'][..., limits, limits].contiguous().cuda())

                        motion_iou_metrics[key](motion_segmentation[..., limits, limits].contiguous(
                        ), motion_labels_seg[..., limits, limits].contiguous().cuda())
                    _viz = False   
                    if _viz:
                        
                        for num,(key, grid) in enumerate(EVALUATION_RANGES.items()):
                            limits = slice(grid[0], grid[1])
                            ml_seg = motion_labels_seg[..., limits, limits].contiguous().squeeze().numpy()
                            m_seg = motion_segmentation[..., limits, limits].contiguous().squeeze().detach().cpu().numpy()
                          
                            
                            instance_ids = np.unique(motion_labels[0]["labels"].detach().cpu().numpy())#[1:]
                            max_instance = np.max(instance_ids)
                            instance_map = dict(zip(instance_ids, instance_ids))
                            instance_colours_dict = {} 
                            
                            
                            plt.figure(0, figsize=(20, 8))
                            plt.title("Test Prediction AttentionThreshold: 65%")
                            n = m_seg.shape[0]
                            fig, axs = plt.subplots(2, n, figsize=(n * 3, 6), gridspec_kw={'hspace': 0.05, 'wspace': 0.3})
                            fig.suptitle("Comparison of Ground Truth Labels and Test Prediction", fontsize=14)
                            #fig.text(0.5, 0.47, 'Test Prediction', ha='center', fontsize=14, fontweight='bold')
                            
                            for k in range(n):
                                color_instance_i, instance_colours = plot_instance_map(
                                    ml_seg[k], instance_map, max_instance)
                                instance_colours_dict.update(instance_colours)
                                axs[0, k].imshow(color_instance_i, vmin=0, vmax=255)
                                #axs[0, k].imshow(motion_labels_seg[k], vmin=0, cmap='gray', vmax=255)
                                axs[0, k].set_title(f"Ground Truth {k + 1}", fontsize=12)
                                axs[0, k].axis('off')
                                
                                color_instance_i, instance_colours = plot_instance_map(
                                    m_seg[k], instance_map, max_instance)
                                instance_colours_dict.update(instance_colours)
                                axs[1, k].imshow(color_instance_i, vmin=0, vmax=255)
                                #axs[1, k].imshow(motion_segmentation[k], cmap='gray', vmin=0, vmax=255)
                                axs[1, k].set_title(f"Prediction {k + 1}", fontsize=12)
                                axs[1, k].axis('off')

                            #plt.show()
                                
                            plt.savefig(
                                rf"/home/niklas/future_instance_prediction_bev/EMT_BEV/viz/test_predictions/sample_{i}_eval_range_{num}.png")
                            
                            plt.close()                    
                else:
                    motion_labels = None

        # update prog_bar
        for _ in range(data_loader.batch_size):
            prog_bar.update()

        # for paper show, combining all results

        if (i + 1) % logging_interval == 0:

            if motion_enable:
                print(
                    '\n[Validation {:04d} / {:04d}]: motion metrics: '.format(motion_eval_count, len(dataset)))

                for key, grid in EVALUATION_RANGES.items():
                    results_str = 'grid = {}: '.format(key)

                    #panoptic_scores = motion_panoptic_metrics[key].compute()
                    iou_scores = motion_iou_metrics[key].compute()

                    results_str += 'iou = {:.3f}, '.format(
                        iou_scores[1].item() * 100)

                    # for panoptic_key, value in panoptic_scores.items():
                    #     results_str += '{} = {:.3f}, '.format(
                    #         panoptic_key, value[1].item() * 100)

                    print(results_str)

            # robust_latencies = latencies[20:]
            # avg_latency = sum(robust_latencies) / len(robust_latencies)
            # print(
            #     ", average forward time = {:.2f}, fps = {:.2f}".format(
            #         avg_latency,
            #         1 / avg_latency,
            #     )
            # )


    print(
        '\n[Validation {:04d} / {:04d}]: motion metrics: '.format(motion_eval_count, len(dataset)))

    for key, grid in EVALUATION_RANGES.items():
        results_str = 'grid = {}: '.format(key)

        #panoptic_scores = motion_panoptic_metrics[key].compute()
        iou_scores = motion_iou_metrics[key].compute()

        results_str += 'iou = {:.3f}, '.format(
            iou_scores[1].item() * 100)

        # for panoptic_key, value in panoptic_scores.items():
        #     results_str += '{} = {:.3f}, '.format(
        #         panoptic_key, value[1].item() * 100)

        print(results_str)

