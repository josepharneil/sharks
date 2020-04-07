import logging
import torch
from detectron2.modeling.meta_arch.retinanet import RetinaNet, permute_all_cls_and_box_to_N_HWA_K_and_concat
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.utils.events import get_event_storage
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss


@META_ARCH_REGISTRY.register()
class RetinaNetOHEM(RetinaNet):
    def __init__(self,cfg):
      super().__init__(cfg)
      self.isOHEM = False
      self.ohem_iter = 250000

    def losses(self, gt_classes, gt_anchors_deltas, pred_class_logits, pred_anchor_deltas):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`RetinaNet.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits` and `pred_anchor_deltas`, see
                :meth:`RetinaNetHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pred_class_logits, pred_anchor_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            pred_class_logits, pred_anchor_deltas, self.num_classes
        )  # Shapes: (N x R, K) and (N x R, 4), respectively.

        gt_classes = gt_classes.flatten()
        gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum().item()
        get_event_storage().put_scalar("num_foreground", num_foreground)
        self.loss_normalizer = (
            self.loss_normalizer_momentum * self.loss_normalizer
            + (1 - self.loss_normalizer_momentum) * num_foreground
        )

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        # print(len(pred_class_logits))

        # logits loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="none",
        ) #/ max(1, self.loss_normalizer)

        # do ohem
        if(self.isOHEM):
            # sum classwise losses for each input and then sort
            ohems,_ = loss_cls.sum(dim=1).sort(descending=True)
            # cut the losses to get the highest 30% ones
            ohems = ohems[:int(round((len(ohems)*0.3)))]
            # finally, sum them
            loss_cls = ohems.sum()
        else:
            # just sum them all
            loss_cls = loss_cls.sum()
        
        # loss_cls = loss_cls.sum() / max(1, self.loss_normalizer)
        loss_cls = loss_cls / max(1, self.loss_normalizer)

        # regression loss
        loss_box_reg = smooth_l1_loss(
            pred_anchor_deltas[foreground_idxs],
            gt_anchors_deltas[foreground_idxs],
            beta=self.smooth_l1_loss_beta,
            reduction="none",
        ) #/ max(1, self.loss_normalizer)
        loss_box_reg = loss_box_reg.sum() / max(1, self.loss_normalizer)

        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}