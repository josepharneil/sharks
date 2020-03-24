from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import shutil
import torch
import os

# 0: COCO-Detection/retinanet_R_50_FPN_1x.yaml
# 1: COCO-Detection/retinanet_R_50_FPN_3x.yaml
# 2: COCO-Detection/retinanet_R_101_FPN_3x.yaml
# 3: COCO-Detection/faster_rcnn_R_50_C4_1x.yaml
# 4: COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml

# modelLink = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"

def CreateCfg(parser,dataset_used,numClasses, baseOutputDir,modelLink,modelOutputFolderName,jobIDOverride=-1):

  # default configuration
  cfg = get_cfg()

  # get the pretrained retinanet model
  cfg.merge_from_file(model_zoo.get_config_file(modelLink))
  # cfg.merge_from_file(model_zoo.get(modelLink,trained=False))#?

  # list of the dataset names for training (registered in datasetcatalog)
  cfg.DATASETS.TRAIN = ("shark_train",)
  # cfg.DATASETS.TRAIN = ("shark_val",)
  # list of the dataset names for testing (registered in datasetcatalog)
  # cfg.DATASETS.TEST = ()
  # cfg.DATASETS.TEST = ("shark_val", )

  # number of data loading threads
  cfg.DATALOADER.NUM_WORKERS = 2

  # locate the pretrained weights
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(modelLink)  # Let training initialize from model zoo

  # number of images per batch
  if(parser.batch_size == 0):
    cfg.SOLVER.IMS_PER_BATCH = 8
    if(dataset_used == "large"):
      cfg.SOLVER.IMS_PER_BATCH = 4
    if(dataset_used == "full"):
      cfg.SOLVER.IMS_PER_BATCH = 4
  else:
    cfg.SOLVER.IMS_PER_BATCH = parser.batch_size


  # learning rate
  if(parser.learning_rate == -1):
    cfg.SOLVER.BASE_LR = 0.005  # pick a good LR
  else:
    cfg.SOLVER.BASE_LR = parser.learning_rate

  # max iterations
  if(parser.max_iter == -1):
    cfg.SOLVER.MAX_ITER = 10000# only if nothing else is entered as an arg
  else:
    cfg.SOLVER.MAX_ITER = parser.max_iter

  # cfg.SOLVER.GAMMA = 0.1#0.1 is default
  # The iteration number to decrease learning rate by GAMMA. 
  # cfg.SOLVER.STEPS = (50000,)#30000 is default
  cfg.SOLVER.STEPS = (30000,)#30000 is default
  # cfg.SOLVER.STEPS = (70000,)#30000 is default

  # Minibatch size PER image - number of regions of interest (ROIs)
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 #lower is faster, default: 512
  # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

  # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 800]]#default doesn't have 800, has 512
  # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 128, 256, 800]]#default doesn't have 800, has 512


  # _C.SOLVER.CHECKPOINT_PERIOD = 5000 #CONSIDER CHANGING THIS
  numberOfCheckpoints = 20
  checkpointPeriod = int(round(cfg.SOLVER.MAX_ITER/numberOfCheckpoints))
  # cfg.SOLVER.CHECKPOINT_PERIOD = checkpointPeriod
  cfg.TEST.EVAL_PERIOD = checkpointPeriod

  # Number of classes
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = numClasses  # only has one class (ballon)
  cfg.MODEL.RETINANET.NUM_CLASSES = numClasses  # only has one class (ballon)

  # def CreateOutputFolder():
  if(jobIDOverride == -1 or jobIDOverride == 0):
    jbName = str(parser.jobid)
  else:
    jbName = str(jobIDOverride)
  foldername = "output_"+jbName
  path = baseOutputDir + modelOutputFolderName + "/" + foldername
  os.makedirs(path, exist_ok=True)
  cfg.OUTPUT_DIR = path

  # CreateOutputFolder()

  # Used in evaluation
  # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
  # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
  # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25  # set the testing threshold for this model
  # cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.25
  cfg.DATASETS.TEST = ("shark_train","shark_val",)


  ###### save the cfg ######
  torch.save(cfg,cfg.OUTPUT_DIR +"/cfg.yaml")

  return cfg