from typing import List
import math
import torch
import torch.nn as nn
import logging
from detectron2.modeling.meta_arch.retinanet import RetinaNet, permute_all_cls_and_box_to_N_HWA_K_and_concat, RetinaNetHead
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.layers import ShapeSpec
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss

class DropoutRetinaNetHead(RetinaNetHead):
  def __init__(self, cfg, input_shape: List[ShapeSpec]):
      super().__init__(cfg, input_shape)
      # fmt: off
      in_channels      = input_shape[0].channels
      num_classes      = cfg.MODEL.RETINANET.NUM_CLASSES
      num_convs        = cfg.MODEL.RETINANET.NUM_CONVS
      prior_prob       = cfg.MODEL.RETINANET.PRIOR_PROB
      num_anchors      = build_anchor_generator(cfg, input_shape).num_cell_anchors
      # fmt: on
      assert (
          len(set(num_anchors)) == 1
      ), "Using different number of anchors between levels is not currently supported!"
      num_anchors = num_anchors[0]

      cls_subnet = []
      bbox_subnet = []
      for _ in range(num_convs):
          # add conv to cls
          cls_subnet.append(
              nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
          )
          # add relu to cls
          cls_subnet.append(nn.ReLU())
          cls_subnet.append(nn.Dropout2d(0.5))
          # do same for relu
          bbox_subnet.append(
              nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
          )
          bbox_subnet.append(nn.ReLU())
          bbox_subnet.append(nn.Dropout2d(0.5))

      self.cls_subnet = nn.Sequential(*cls_subnet)
      self.bbox_subnet = nn.Sequential(*bbox_subnet)
      self.cls_score = nn.Conv2d(
          in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
      )
      self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)

      # Initialization
      for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
          for layer in modules.modules():
              if isinstance(layer, nn.Conv2d):
                  torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                  torch.nn.init.constant_(layer.bias, 0)

      # Use prior in model initialization to improve stability
      bias_value = -math.log((1 - prior_prob) / prior_prob)
      torch.nn.init.constant_(self.cls_score.bias, bias_value)

# @META_ARCH_REGISTRY.register()
class DropoutRetinaNet(RetinaNet):
  def __init__(self,cfg):
    super().__init__(cfg)
    backbone_shape = self.backbone.output_shape()
    feature_shapes = [backbone_shape[f] for f in self.in_features]
    self.head = DropoutRetinaNetHead(cfg, feature_shapes).to(self.device)

    self.to(self.device)