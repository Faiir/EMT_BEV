import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
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

from ..motion_modules import ResFuturePrediction, ResFuturePredictionV2
from ._base_motion_head import BaseMotionHead
from timeit import default_timer as timer
import pdb
from torch.profiler import record_function


@HEADS.register_module()
class IterativeFlow(BaseMotionHead):
    def __init__(
        self,
        detach_state=True,
        n_gru_blocks=1,
        using_v2=False,
        flow_warp=True,
        **kwargs,
    ):
        super(IterativeFlow, self).__init__(**kwargs)
        print("IterativeFlow")
        self.logger = logging.getLogger("timelogger")
        if using_v2:
            self.future_prediction = ResFuturePredictionV2(
                in_channels=self.in_channels,
                latent_dim=self.prob_latent_dim,
                n_future=self.n_future,
                detach_state=detach_state,
                n_gru_blocks=n_gru_blocks,
                flow_warp=flow_warp,
            )
        else:
            self.future_prediction = ResFuturePrediction(
                in_channels=self.in_channels,
                latent_dim=self.prob_latent_dim,
                n_future=self.n_future,
                detach_state=detach_state,
                n_gru_blocks=n_gru_blocks,
                flow_warp=flow_warp,
            )

    def forward(self, bevfeats, targets=None, noise=None):
        """
        the forward process of motion head:
        1. get present & future distributions
        2. iteratively get future states with ConvGRU
        3. decode present & future states with the decoder heads
        """
        torch.cuda.synchronize()
        start_f = timer()
        with record_function("iterative_flow_total"):
            bevfeats = bevfeats[0]
            if self.training or self.posterior_with_label:
                (
                    self.training_labels,
                    future_distribution_inputs,
                ) = self.prepare_future_labels(targets)
            else:
                future_distribution_inputs = None

            res = {}
            if self.n_future > 0:
                with record_function("iterative_flow_part1"):
                    present_state = bevfeats.unsqueeze(dim=1).contiguous()
                    self.logger.debug(
                        "Temp present_state shape " + str(present_state.shape)
                    )

                    # sampling probabilistic distribution
                    torch.cuda.synchronize()
                    start = timer()
                    sample, output_distribution = self.distribution_forward(
                        present_state, future_distribution_inputs, noise
                    )
                    torch.cuda.synchronize()
                    end = timer()
                    t_distribution_forward = (end - start) * 1000
                    self.logger.debug(
                        "Temp output_distribution shape "
                        + str(list(output_distribution.keys()))
                    )
                    self.logger.debug("Temp sample shape " + str(sample.shape))
                    self.logger.debug(
                        "Temp distribution_forward " + str(t_distribution_forward)
                    )

                    b, _, _, h, w = present_state.shape
                    hidden_state = present_state[:, 0]

                    self.logger.debug(
                        "Temp hidden_state shape " + str(hidden_state.shape)
                    )

                future_states = self.future_prediction(sample, hidden_state)
                self.logger.debug(
                    "Temp future_states shape " + str(future_states.shape)
                )
                future_states = torch.cat([present_state, future_states], dim=1)
                self.logger.debug(
                    "Temp future_states shape " + str(future_states.shape)
                )
                # flatten dimensions of (batch, sequence)
                batch, seq = future_states.shape[:2]
                flatten_states = future_states.flatten(0, 1)

                if self.training:
                    res.update(output_distribution)

                for task_key, task_head in self.task_heads.items():
                    res[task_key] = task_head(flatten_states).view(batch, seq, -1, h, w)
            else:
                b, _, h, w = bevfeats.shape
                for task_key, task_head in self.task_heads.items():
                    res[task_key] = task_head(bevfeats).view(b, 1, -1, h, w)

            torch.cuda.synchronize()
            end_f = timer()
            t_IterativeFlow = (end_f - start_f) * 1000
            self.logger.debug(" t_IterativeFlow " + str(t_IterativeFlow))
            # self.logger.debug("Temp future_states shape " + str(future_states.shape))
        return res
