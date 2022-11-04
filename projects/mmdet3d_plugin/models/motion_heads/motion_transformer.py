import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from mmdet3d.models.builder import HEADS
from ..dense_heads.base_taskhead import BaseTaskHead
from ..dense_heads.loss_utils import (
    MotionSegmentationLoss,
    SpatialRegressionLoss,
    ProbabilisticLoss,
    GaussianFocalLoss,
    SpatialProbabilisticLoss,
)

from ...datasets.utils.geometry import cumulative_warp_features_reverse
from ...datasets.utils.instance import predict_instance_segmentation_and_trajectories
from ...datasets.utils.warper import FeatureWarper

from ...visualize import Visualizer
from ._base_motion_head import BaseMotionHead

from ..deformable_detr_modules import build_backbone, build_seg_detr, build_deforamble_transformer, build_position_encoding
from mmcv.runner import auto_fp16, force_fp32
from torch.profiler import record_function
import pdb


@HEADS.register_module()
class Motion_DETR(BaseMotionHead):
    def __init__(self,DETR_ARGS ,
                 **kwargs):
        super(Motion_DETR, self).__init__(**kwargs)
        self.logger = logging.getLogger("timelogger")

        backbone = build_backbone(DETR_ARGS)
        b
    
    def forward(self,):
        pass 
    