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

from ..deformable_detr_modules import build_MaskHeadSmallConv,build_detr,build_backbone, build_seg_detr, build_deforamble_transformer, build_position_encoding
from mmcv.runner import auto_fp16, force_fp32
from torch.profiler import record_function
import pdb


def build_output_convs(output_dict={1,2,2,})


@HEADS.register_module()
class Motion_DETR(BaseMotionHead):
    def __init__(self,DETR_ARGS ,
                 receptive_field=3,
                 n_future=0,
                 future_discount = 0.95,
                 grid_conf=None,
                class_weights=None,
                use_topk=True,
                topk_ratio=0.25,
                ignore_index=255,
                posterior_with_label=False,
                sample_ignore_mode="all_valid",
                using_focal_loss=False,
                focal_cfg=dict(type="GaussianFocalLoss", reduction="none"),
                loss_weights=None,
                 #DETR ARGS HERE
                 
                 
                 #DETR END  ARGS HERE
                train_cfg=None,
                test_cfg=None,
                init_cfg=None,
                 **kwargs):
        super(Motion_DETR, self).__init__(**kwargs)
        self.logger = logging.getLogger("timelogger")
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.receptive_field = receptive_field
        self.n_future = n_future
        
        self.backbone = build_backbone(backbone=backbone, layers=[
                                       2, 2, 2, 2], return_feature_layers=True, position_embedding=position_embedding, num_pos_feats=num_pos_feats)
        
        
        self.future_heads = build_output_convs()
        self.mask_conv = build_MaskHeadSmallConv(hidden_dim=hidden_dim,nheads=nheads,fpns=fpns)
        self.transformer = build_deforamble_transformer(hidden_dim, nheads, enc_layers, dec_layers,
                                                        dim_feedforward, dropout_transformer, activation,
                                                        num_feature_levels, dec_n_points, enc_n_points,
                                                        num_queries)
        
        self.DETR = build_detr(
            self.backbone, self.transformer, num_classes, num_queries, num_feature_levels)
        self.DETR_SEG = build_seg_detr(
            self.DETR, mall_resnet=small_resnet, output_convs=output_convs)
        # loss functions
        # 1. loss for foreground segmentation
        self.seg_criterion = MotionSegmentationLoss(
            class_weights=torch.tensor(),
            use_top_k=use_topk,
            top_k_ratio=topk_ratio,
            future_discount=future_discount,
        )

        # 2. loss for instance center heatmap
        self.reg_instance_center_criterion = SpatialRegressionLoss(
            norm=2,
            future_discount=future_discount,
        )

        self.cls_instance_center_criterion = GaussianFocalLoss(
            focal_cfg=focal_cfg,
            ignore_index=ignore_index,
            future_discount=future_discount,
        )

        # 3. loss for instance offset
        self.reg_instance_offset_criterion = SpatialRegressionLoss(
            norm=1,
            future_discount=future_discount,
            ignore_index=ignore_index,
        )

        # 4. loss for instance flow
        self.reg_instance_flow_criterion = SpatialRegressionLoss(
            norm=1,
            future_discount=future_discount,
            ignore_index=ignore_index,
        )

        self.probabilistic_loss = ProbabilisticLoss(foreground=self.prob_on_foreground)

        # pass prediction heads here -> maybe take out the seg-head and pass it as object to the builder 
        self.deformable_detr = build_seg_detr(DETR_ARGS, self.transformer, self.backbone)
        
        self.loss_weights = loss_weights
        self.ignore_index = ignore_index
        self.using_focal_loss = using_focal_loss

        self.visualizer = Visualizer(out_dir="train_visualize")
        self.warper = FeatureWarper(grid_conf=grid_conf)
    


        
    def forward(self, bevfeats, targets=None, noise=None):
        """
        the forward process of motion head:
        1. get present & future distributions
        2. iteratively get future states with ConvGRU
        3. decode present & future states with the decoder heads
        """
        bevfeats = bevfeats[0]
        bev_mask = None #TODO


        res = {}
        if self.n_future > 0:
            with record_function("Motion Prediction distribution forward"):

                


                for task_key, task_head in self.task_heads.items():
                    res[task_key] = task_head(flatten_states).view(batch, seq, -1, h, w)
        else:
            b, _, h, w = bevfeats.shape
            for task_key, task_head in self.task_heads.items():
                res[task_key] = task_head(bevfeats).view(b, 1, -1, h, w)

        return res

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        if "instance_center" in self.task_heads:
            self.task_heads["instance_center"][-1].bias.data.fill_(self.init_bias)



    def prepare_future_labels(self, batch):
        labels = {}
        #future_distribution_inputs = []

        segmentation_labels = batch["motion_segmentation"]
        instance_center_labels = batch["instance_centerness"]
        instance_offset_labels = batch["instance_offset"]
        instance_flow_labels = batch["instance_flow"]
        gt_instance = batch["motion_instance"]
        future_egomotion = batch["future_egomotion"]
        bev_transform = batch.get("aug_transform", None)
        labels["img_is_valid"] = batch.get("img_is_valid", None)

        if bev_transform is not None:
            bev_transform = bev_transform.float()

        segmentation_labels = (
            self.warper.cumulative_warp_features_reverse(
                segmentation_labels.float().unsqueeze(2),
                future_egomotion[:, (self.receptive_field - 1) :],
                mode="nearest",
                bev_transform=bev_transform,
            )
            .long()
            .contiguous()
        )
        print(f"Seg labels shape: {segmentation_labels.shape}")
        labels["segmentation"] = segmentation_labels
        #future_distribution_inputs.append(segmentation_labels)

        # Warp instance labels to present's reference frame
        gt_instance = (
            self.warper.cumulative_warp_features_reverse(
                gt_instance.float().unsqueeze(2),
                future_egomotion[:, (self.receptive_field - 1) :],
                mode="nearest",
                bev_transform=bev_transform,
            )
            .long()
            .contiguous()[:, :, 0]
        )
        labels["instance"] = gt_instance
        print(f"gt_instance shape: {gt_instance.shape}")
        instance_center_labels = self.warper.cumulative_warp_features_reverse(
            instance_center_labels,
            future_egomotion[:, (self.receptive_field - 1) :],
            mode="nearest",
            bev_transform=bev_transform,
        ).contiguous()
        labels["centerness"] = instance_center_labels
        print(f"instance_center_labels shape: {instance_center_labels.shape}")
        instance_offset_labels = self.warper.cumulative_warp_features_reverse(
            instance_offset_labels,
            future_egomotion[:, (self.receptive_field - 1) :],
            mode="nearest",
            bev_transform=bev_transform,
        ).contiguous()
        labels["offset"] = instance_offset_labels


        instance_flow_labels = self.warper.cumulative_warp_features_reverse(
            instance_flow_labels,
            future_egomotion[:, (self.receptive_field - 1) :],
            mode="nearest",
            bev_transform=bev_transform,
        ).contiguous()
        labels["flow"] = instance_flow_labels
        
        print(f"instance_flow_labels shape: {instance_flow_labels.shape}")

        # self.visualizer.visualize_motion(labels=labels)
        # pdb.set_trace()

        return labels

    @force_fp32(apply_to=("predictions"))
    def loss(self, predictions, targets=None):
        print("Loss base motion head")
        loss_dict = {}

        """
        prediction dict:
            'segmentation': 2,
            'instance_center': 1,
            'instance_offset': 2,
            'instance_flow': 2,
        """

        for key, val in self.training_labels.items():
            self.training_labels[key] = val.float()

        frame_valid_mask = self.training_labels["img_is_valid"].bool()
        past_valid_mask = frame_valid_mask[:, : self.receptive_field]
        future_frame_mask = frame_valid_mask[:, (self.receptive_field - 1) :]

        if self.sample_ignore_mode is "all_valid":
            # only supervise when all 7 frames are valid
            batch_valid_mask = frame_valid_mask.all(dim=1)
            future_frame_mask[~batch_valid_mask] = False
            prob_valid_mask = batch_valid_mask

        elif self.sample_ignore_mode is "past_valid":
            # only supervise when past 3 frames are valid
            past_valid = torch.all(past_valid_mask, dim=1)
            future_frame_mask[~past_valid] = False
            prob_valid_mask = past_valid

        elif self.sample_ignore_mode is "none":
            prob_valid_mask = frame_valid_mask.any(dim=1)

        # segmentation
        loss_dict["loss_motion_seg"] = self.seg_criterion(
            predictions["segmentation"],
            self.training_labels["segmentation"].long(),
            frame_mask=future_frame_mask,
        )

        # instance centerness, but why not focal loss
        if self.using_focal_loss:
            loss_dict["loss_motion_centerness"] = self.cls_instance_center_criterion(
                predictions["instance_center"],
                self.training_labels["centerness"],
                frame_mask=future_frame_mask,
            )
        else:
            loss_dict["loss_motion_centerness"] = self.reg_instance_center_criterion(
                predictions["instance_center"],
                self.training_labels["centerness"],
                frame_mask=future_frame_mask,
            )

        # instance offset
        loss_dict["loss_motion_offset"] = self.reg_instance_offset_criterion(
            predictions["instance_offset"],
            self.training_labels["offset"],
            frame_mask=future_frame_mask,
        )

        if self.n_future > 0:
            # instance flow
            loss_dict["loss_motion_flow"] = self.reg_instance_flow_criterion(
                predictions["instance_flow"],
                self.training_labels["flow"],
                frame_mask=future_frame_mask,
            )

            if self.probabilistic_enable:
                loss_dict["loss_motion_prob"] = self.probabilistic_loss(
                    predictions,
                    foreground_mask=self.training_labels["segmentation"],
                    batch_valid_mask=prob_valid_mask,
                )

        for key in loss_dict:
            loss_dict[key] *= self.loss_weights.get(key, 1.0)

        return loss_dict

    def inference(self, predictions):
        # [b, s, num_cls, h, w]
        seg_prediction = torch.argmax(predictions["segmentation"], dim=2, keepdims=True)

        if self.using_focal_loss:
            predictions["instance_center"] = torch.sigmoid(
                predictions["instance_center"]
            )

        pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
            predictions,
            compute_matched_centers=False,
        )

        return seg_prediction, pred_consistent_instance_seg