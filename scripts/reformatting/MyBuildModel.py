import torch
import detectron2
import MyVGG
import MyYOLO
import RetinaNetOHEM
import DropoutRetinaNet
from detectron2.modeling import build_model
import logging

def my_build_model(cfg):
  """
  Returns:
      torch.nn.Module:

  It now calls :func:`detectron2.modeling.build_model`.
  Overwrite it if you'd like a different model.
  """
  # handle yolo and vgg
  if(cfg.MODEL.META_ARCHITECTURE == "VGG19_BN"):
      model = MyVGG.Create_VGG(cfg.MODEL.RETINANET.NUM_CLASSES)
  elif(cfg.MODEL.META_ARCHITECTURE == "YOLOV3"):
      model = MyYOLO.Create_YOLO(cfg.MODEL.RETINANET.NUM_CLASSES)
  elif(cfg.MODEL.META_ARCHITECTURE == "RetinaNetOHEM"):
      model = RetinaNetOHEM.RetinaNetOHEM(cfg)
  elif(cfg.MODEL.META_ARCHITECTURE == "DropoutRetinaNet"):
      model = DropoutRetinaNet.DropoutRetinaNet(cfg)
  elif(cfg.MODEL.META_ARCHITECTURE == "DropoutRetinaNet2"):
      model = DropoutRetinaNet.DropoutRetinaNet2(cfg)
  else:
      model = build_model(cfg)
      logger = logging.getLogger(__name__)
      logger.info("Model:\n{}".format(model))
      
  return model