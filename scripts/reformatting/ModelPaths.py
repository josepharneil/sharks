def GetModelLinks(model_index):
  modelLink = ""
  modelOutputFolderName = ""
  if(model_index == 0):
    modelLink = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"
    modelOutputFolderName = "retinanet_R_50_FPN_1x"
    meta_arch_override = None
  elif(model_index == 1):
    modelLink = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
    modelOutputFolderName = "retinanet_R_50_FPN_3x"
    meta_arch_override = None
  elif(model_index == 2):
    modelLink = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
    modelOutputFolderName = "retinanet_R_101_FPN_3x"
    meta_arch_override = None
  elif(model_index == 3):
    modelLink = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
    modelOutputFolderName = "faster_rcnn_X_101_32x8d_FPN_3x"
    meta_arch_override = None
  elif(model_index == 4):
    modelLink = "VGG19_BN"
    modelOutputFolderName = "VGG19_BN"
    meta_arch_override = None
  elif(model_index == 5):
    modelLink = "YOLOV3"
    modelOutputFolderName = "YOLOV3"
    meta_arch_override = None
  elif(model_index == 6):
    modelLink = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
    modelOutputFolderName = "retinanet_R_101_FPN_3x_OHEM"
    meta_arch_override = "RetinaNetOHEM"
  elif(model_index == 7):
    modelLink = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
    modelOutputFolderName = "retinanet_R_101_FPN_3x_DROPOUT"
    meta_arch_override = "DropoutRetinaNet"
  elif(model_index == 8):
    modelLink = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    modelOutputFolderName = "faster_rcnn_R_101_FPN_3x"
    meta_arch_override = None
  else:
    raise ValueError("No such model index:", model_index)

  return modelLink,modelOutputFolderName,meta_arch_override
