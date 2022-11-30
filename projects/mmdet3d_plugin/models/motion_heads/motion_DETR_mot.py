import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from mmdet3d.models.builder import HEADS


from ...datasets.utils.geometry import cumulative_warp_features_reverse
from ...datasets.utils.instance import predict_instance_segmentation_and_trajectories
from ...datasets.utils.warper import FeatureWarper
from ...visualize import Visualizer


from ..deformable_detr_modules import MLP, MaskHeadSmallConvIFC
from mmdet.core import build_assigner


from mmcv.runner import auto_fp16, force_fp32
from torch.profiler import record_function
import pdb

from mmcv.runner import BaseModule


@HEADS.register_module()
class Motion_DETR_MOT(BaseModule):
    def __init__(self ,
                 receptive_field=3,
                 n_future=0,
                 future_discount = 0.95,
                 grid_conf=None,
                 class_weights=[1.0, 2.0],
                hidden_dim=512, 
                nheads=8,
                use_topk=True,
                
                ignore_index=255,
                num_queries=300,
                #posterior_with_label=False, TODO 
                sample_ignore_mode="all_valid", # TODO 
                loss_weights=None,
                matcher_config={
                     "cost_class": 1,
                     "cost_dice": 3.0,
                     "mask_weight":3.0
                 },
                criterion_config={
                    "num_classes": 80,
                    "weight_dict": {"loss_ce": 1, "loss_mask": 3.0,
                                    "loss_dice": 3.0},
                    "eos_coef": 0.1,
                    "losses": ["labels", "masks", "cardinality"],
                },
                dec_layers=1,
                task_dict=None,
                train_cfg=None,
                test_cfg=None,
                init_cfg=dict(type="Kaiming", layer="Conv2d"),
                 **kwargs):
        super(Motion_DETR_MOT, self).__init__(**kwargs)
        self.logger = logging.getLogger("timelogger")
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.receptive_field = receptive_field
        self.n_future = n_future
        

        self.task_heads = nn.ModuleDict()
        
        matcher = build_assigner(matcher_config)
        

        # matcher = HungarianMatcherIFC(
        #     cost_class=1,
        #     cost_dice=dice_weight,
        #     num_classes=criterion_config["num_classes"],
        
        # )
        weight_dict = criterion_config["weight_dict"]
        if dec_layers > 1:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in weight_dict.items()})
                
            weight_dict.update(aux_weight_dict)
        self.num_classes = criterion_config["num_classes"]
        criterion = SetCriterion(
            self.num_classes, matcher=matcher, weight_dict=criterion_config[
                "weight_dict"], eos_coef=criterion_config["eos_coef"], losses=criterion_config["losses"],
            n_future=n_future
        )
        self.class_mlps = []
        for _ in range(self.n_future):
            self.class_mlps.append(MLP(hidden_dim, hidden_dim,
                              output_dim=self.num_classes + 1, num_layers=2))
            
            
        fpn_dims_input = [512, 256, 128, 64]
        fpn_dims = [256, 256, 256, 256]
        
        self.project_convs = []
        for _in,out in zip(fpn_dims_input,fpn_dims):
            self.project_convs.append(nn.Conv2d(_in, out, 3, padding=1))
        
        
        self.mask_conv = MaskHeadSmallConvIFC(
            hidden_dim, fpn_dims, hidden_dim)


        self.visualizer = Visualizer(out_dir="train_visualize")
        self.warper = FeatureWarper(grid_conf=grid_conf)


        #self.bev_projection = nn.Conv2d(in_channels=64,out_channels=64,kernel=1,padding=0)


    def _set_aux_loss(self, outputs_class, outputs_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_masks': b}
                for a, b in zip(outputs_class[:-1], outputs_masks[:-1])]

    def forward(self,  hs, reference, seg_memory, seg_mask, pyramid_bev_feats, targets=None, noise=None):
        """
        the forward process of motion head:
        1. get present & future distributions
        2. iteratively get future states with ConvGRU
        3. decode present & future states with the decoder heads
        """
        input_projections = []
        for c,proj_conv in enumerate(self.project_convs):
            input_projections.append(proj_conv(pyramid_bev_feats[c]))

        outputs_masks = self.mask_head(
            pyramid_bev_feats[-1], seg_memory, input_projections, hs)

        outputs_class = []
        for class_mlp in self.class_mlps:
            outputs_class.append(class_mlp(hs))

        outputs_class = torch.stack(outputs_class)

        out = {'pred_logits': outputs_class[-1]}
        out.update({'pred_masks': outputs_masks[-1]})

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_masks)

        return out

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        if "instance_center" in self.task_heads:
            self.task_heads["instance_center"][-1].bias.data.fill_(self.init_bias)


    def prepare_future_labels(self, batch, mask_stride=2, match_stride=2):
        #segmentation_labels = batch["motion_segmentation"][0]
        gt_instance = batch["motion_instance"][0]
        future_egomotion = batch["future_egomotions"][0]
        batch_size = len(gt_instance)
        labels = {}

        bev_transform = batch.get("aug_transform", None)
        labels["img_is_valid"] = batch.get("img_is_valid", None)

        if bev_transform is not None:
            bev_transform = bev_transform.float()

        # Warp instance labels to present's reference frame
        gt_instance = (
            self.warper.cumulative_warp_features_reverse(
                gt_instance.float().unsqueeze(2),
                future_egomotion[:, (self.receptive_field - 1):],
                mode="nearest",
                bev_transform=bev_transform,
            )
            .long()
            .contiguous()[:, :, 0]
        )
        # better solution by abdur but unsure how to make it work with the rest of the code specifcally maxID since it can be diffferent for batches
        # temp = torch.arange(MaxID).unsqueeze(0).repeat(B, 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # gt_masks_ifc_dim  = (temp== Target.unsqueeze(1)).float()
        target_list = []
        for b in range(batch_size):
            gt_list = []
            ids = len(gt_instance[b].unique())
            for _id in range(ids):
                test_bool = torch.where(gt_instance[b] == _id, 1., 0.)
                gt_list.append(test_bool)

            segmentation_labels = torch.stack(gt_list, dim=0)

            #segmentation_labels = torch.stack(gt_batch_instances_list,dim=0)
            o_h, o_w = segmentation_labels[-2:]
            l_h, l_w = math.ceil(o_h/mask_stride), math.ceil(o_w/mask_stride)
            m_h, m_w = math.ceil(o_h/match_stride), math.ceil(o_w/match_stride)

            gt_masks_for_loss = F.interpolate(segmentation_labels, size=(
                l_h, l_w), mode="bilinear", align_corners=False)
            gt_masks_for_match = F.interpolate(segmentation_labels, size=(
                m_h, m_w), mode="bilinear", align_corners=False)

            # labels only continous for clip - this is much more of an tracking id as every class is a vehicle anyways # TODO make work with other types of superclasses other then vehicle
            ids = gt_instance[b].unique()
            target_list.append({"labels": ids, "masks": gt_masks_for_loss,
                               "match_masks": gt_masks_for_match, "gt_motion_instance": gt_instance[b]})
        return target_list, future_egomotion[:, (self.receptive_field - 1):]
        

    @force_fp32(apply_to=("predictions"))
    def loss(self, predictions, targets=None):
        print("Loss base motion head")
        loss_dict = {}

        target_list,_ = self.prepare_future_labels(targets)
        loss_dict = self.criterion(predictions, target_list)

        for key in loss_dict:
            loss_dict[key] *= self.loss_weights.get(key, 1.0)

        return loss_dict

    def inference(self, predictions): #TODO
        # [b, s, num_cls, h, w]
        mask_cls = predictions["pred_logits"]
        mask_pred = predictions["pred_masks"]

        # For each mask we assign the best class or the second best if the best on is `no_object`.
        _idx = self.num_classes + 1
        mask_cls = F.softmax(mask_cls, dim=-1)[:, :_idx]
        scores, labels = mask_cls.max(-1)

        valid = (labels < self.num_classes)
        scores = scores[valid]
        labels = labels[valid]
        mask_cls = mask_cls[valid]
        mask_pred = mask_pred[valid]

        results = "todp"
        # results = Instances(image_size)
        # results.scores = scores
        # results.pred_classes = labels
        # results.cls_probs = mask_cls
        # results.pred_masks = mask_pred

        return results

        # predictions["segmentation"] = torch.nn.interpolate(predictions["segmentation"][:, None], size=self.training_labels["segmentation"].shape[-2:],
        #                                                    mode="bilinear", align_corners=False)
        # seg_prediction = torch.argmax(predictions["segmentation"], dim=2, keepdims=True)

        # #non max suppression
        # pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
        #     predictions,
        #     compute_matched_centers=False,
        # )

        #return seg_prediction, pred_consistent_instance_seg
    
    
@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_coef(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    N, M = len(inputs), len(targets)
    inputs = inputs.flatten(1).unsqueeze(1).expand(-1, M, -1)
    targets = targets.flatten(1).unsqueeze(0).expand(N, -1, -1)

    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    coef = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        coef = alpha_t * coef

    return coef.mean(2)


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class SetCriterion(nn.Module):
    """ This class computes the loss for IFC.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth masks and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and mask)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, n_future):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.n_future = n_future
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_masks, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_masks]
        """
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J]
                                     for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(
            1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - \
                accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_masks):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) !=
                     pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_masks, h, w]
        """
        assert "pred_masks" in outputs

        idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"][idx]

        target_masks = torch.cat(
            [t['masks'][i] for t, (_, i) in zip(targets, indices)]).to(src_masks)

        n, t = src_masks.shape[:2]
        t_h, t_w = target_masks.shape[-2:]

        src_masks = F.interpolate(src_masks, size=(
            t_h, t_w), mode="bilinear", align_corners=False)

        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_masks),
            "loss_dice": dice_loss(src_masks, target_masks, num_masks),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)
                              for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_masks, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k,
                               v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target masks accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / 1, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(
                loss, outputs_without_aux, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_masks, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
