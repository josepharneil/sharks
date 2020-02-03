# -*- coding: utf-8 -*-
"""shark-initialseti.pynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oUGEC-RPu9DxnDLidVM2su46DVpjjnpV

# Installs and Imports
"""

# install dependencies:
# (use +cu100 because colab is on CUDA 10.0)
# !pip install -U torch==1.4+cu100 torchvision==0.5+cu100 -f https://download.pytorch.org/whl/torch_stable.html 
# !pip install cython pyyaml==5.1
# !pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
import torch, torchvision
# torch.__version__
# !gcc --version
# opencv is pre-installed on colab

# install detectron2:
# !git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# !pip install -e detectron2_repo

# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
# from google.colab.patches import cv2_imshow
import math

import cv2
import os
import numpy as np
import json
from detectron2.structures import BoxMode

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import argparse

# from google.colab import drive
# drive.mount('/content/drive')

"""# Directories"""

trainDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/small_set/photos/train/"
valDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/small_set/photos/val/"
imageDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/small_set/photos/images/"
sourceJsonDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/small_set/photos/data.json"
baseDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/small_set/photos/"
baseOutputDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/outputs/"


"""# Parser"""

parser = argparse.ArgumentParser(
    description="Train over small shark set",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
  "-lr",
  "--learning-rate",
  default=-1,
  type=float,
  help="Base learning rate used by the model"
)
parser.add_argument(
  "-m",
  "--model",
  default=-1,
  type=int,
  help="Model used"
)
parser.add_argument(
  "-i",
  "--max-iter",
  default=-1,
  type=int,
  help="Max number of iterations in training"
)




"""# Dataset

Construct Dictionary of SharkIDs to Classes
"""

# Registered in dataset loading
def getSharkDicts(trainVal):
  if(trainVal == "train"):
    print("Getting shark train dicts")
    return getSharkTrainDicts()
  if(trainVal == "val"):
    print("Getting shark val dicts")
    return getSharkValDicts()

# Called by getSharkDicts
def getSharkTrainDicts():
  return torch.load(trainDirectory+"sharkTrainDicts.pt")

# Called by getSharkDicts
def getSharkValDicts():
  return torch.load(valDirectory+"sharkValDicts.pt")

def getSharkClassDictionary():
  return torch.load(baseDirectory+"SharkClassDictionary.pt")

def getClassList():
  return torch.load(baseDirectory+"ClassList.pt")


SharkClassDictionary = getSharkClassDictionary()
ClassList = getClassList()

# Enter into dataset and metadata catalogues
from detectron2.data import DatasetCatalog, MetadataCatalog
for d in ["train", "val"]:
  # Register shark_train and shark_val
  DatasetCatalog.register("shark_" + d, lambda d=d: getSharkDicts(d))
  MetadataCatalog.get("shark_" + d).set(thing_classes=ClassList)

shark_metadata = MetadataCatalog.get("shark_train")

"""Visualise to check this worked"""

# dataset_dicts = getSharkDicts("/content/drive/My Drive/sharkdata/train")
dataset_dicts = getSharkTrainDicts() #getSharkDicts("/content/drive/My Drive/sharkdata/all_data/train")
# dataset_dicts = getSharkValDicts()
# print("Done")
# for dictionary in random.sample(dataset_dicts, 12):
# # for dictionary in dataset_dicts:
#   scl = 0.2
#   img = cv2.imread(dictionary["file_name"])
#   visualizer = Visualizer(img[:, :, ::-1], metadata=shark_metadata, scale=scl)
#   vis = visualizer.draw_dataset_dict(dictionary)
#   # Print the class ID
#   classID = ((dictionary["annotations"])[0])["category_id"]
#   print()
#   print(ClassList[classID])
#   print(dictionary["file_name"])

#   # cv2_imshow(vis.get_image()[:, :, ::-1])

#   #crop
#   bbox = ((dictionary["annotations"])[0])["bbox"]
#   print(bbox) #"bbox": [xmin,ymin,xmax,ymax],
#   xmin,ymin,xmax,ymax = bbox
#   w = xmax-xmin
#   h = ymax-ymin
#   print(w,h)
#   # cropT = detectron2.data.transforms.CropTransform(xmin-5,ymin-5,w+5,h+5)
#   cropT = detectron2.data.transforms.CropTransform(math.floor((xmin-15)*scl),math.floor((ymin-15)*scl),math.floor((w+20)*scl),math.floor((h+20)*scl))
#   cropIm = cropT.apply_image(vis.get_image()[:, :, ::-1])
#   cv2_imshow(cropIm)

  # break

"""# Dataset Mapping"""

# Building my own data loader
from detectron2.data import build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
import copy

def mapper(dataset_dict):
  # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
  # Create a copy of the dataset dict
  dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
  # Read in the image
  image = utils.read_image(dataset_dict["file_name"], format="BGR")

  # cropping
  # Get boundingbox
  bbox = ((dataset_dict["annotations"])[0])["bbox"]
  # print(bbox)
  xmin,ymin,xmax,ymax = bbox
  w = xmax-xmin
  h = ymax-ymin

  # print(image.shape)

  cropT = T.CropTransform(xmin-15,ymin-15,w+50,h+50)
  image = cropT.apply_image(image)

  dataset_dict["height"] = h+15+50
  dataset_dict["width"] = w+15+50
  # cv2_imshow(image)

  # image, transforms = T.apply_transform_gens([cropT], image)

  # print()
  # print(T.Resize((800, 800)).dtype)
  # print()

  # T.Transform()

  # print(cropIm.shape)

  # image, transforms = T.apply_transform_gens([T.Resize((800, 800))], image)
  # image, transforms = T.apply_transform_gens([T.CropTransform(math.floor((xmin-2)),math.floor((ymin-2)),math.floor((w+5)),math.floor((h+5)))], image)
  # image, transforms = T.apply_transform_gens([utils.gen_crop_transform_with_instance((h,w), (dataset_dict["height"],dataset_dict["width"]), (dataset_dict["annotations"])[0]  )], image)
  # gen_crop_transform_with_instance((h,w), (dataset_dict["height"],dataset_dict["width"]), dataset_dict)
  # print(image.size)
  # transforms = T.TransformList([cropT])
  transforms = T.TransformList([cropT])

  
  # image, tfms = T.apply_transform_gens([T.ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style='choice'), T.RandomFlip()], image)
  # image, tfms = T.apply_transform_gens([T.RandomFlip()], image)
  image, tfms = T.apply_transform_gens([T.RandomFlip()], image)
  transforms = transforms + tfms

  # print(image.shape)

  # print(image.shape)
  dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

  # cv2_imshow(image)

  annos = \
  [
      utils.transform_instance_annotations(obj, transforms, image.shape[:2])
      for obj in dataset_dict.pop("annotations")
      if obj.get("iscrowd", 0) == 0
  ]
  instances = utils.annotations_to_instances(annos, image.shape[:2])
  dataset_dict["instances"] = utils.filter_empty_instances(instances)

  return dataset_dict

from detectron2.engine import DefaultTrainer

class Trainer(DefaultTrainer):
    # @classmethod
    # def build_evaluator(cls, cfg, dataset_name):
    #     output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    #     evaluators = [COCOEvaluator(dataset_name, cfg, True, output_folder)]
    #     if cfg.MODEL.DENSEPOSE_ON:
    #         evaluators.append(DensePoseCOCOEvaluator(dataset_name, True, output_folder))
    #     return DatasetEvaluators(evaluators)

    # @classmethod
    # def build_test_loader(cls, cfg, dataset_name):
    #     return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=mapper)

"""# Training"""

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import shutil



# 0: COCO-Detection/retinanet_R_50_FPN_1x.yaml
# 1: COCO-Detection/retinanet_R_50_FPN_3x.yaml
# 2: COCO-Detection/retinanet_R_101_FPN_3x.yaml
# 3: COCO-Detection/faster_rcnn_R_50_C4_1x.yaml
# 4: COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml

# modelLink = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"
modelLink = ""
if(parser.parse_args().model == 0):
  modelLink = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"
elif(parser.parse_args().model == 1):
  modelLink = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
elif(parser.parse_args().model == 2):
  modelLink = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
elif(parser.parse_args().model == 3):
  modelLink = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"
elif(parser.parse_args().model == 4):
  modelLink = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
else:
  modelLink = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"


# default configuration
cfg = get_cfg()

# get the pretrained retinanet model
cfg.merge_from_file(model_zoo.get_config_file(modelLink))
# cfg.merge_from_file(model_zoo.get(modelLink,trained=False))

# list of the dataset names for training (registered in datasetcatalog (?))
cfg.DATASETS.TRAIN = ("shark_train",)
# list of the dataset names for testing (registered in datasetcatalog (?))
cfg.DATASETS.TEST = ()

##cropping
# cfg.INPUT.CROP({"ENABLED": False})
# cfg.INPUT.CROP.ENABLED = True
##

# number of data loading threads
cfg.DATALOADER.NUM_WORKERS = 2

# locate the pretrained weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(modelLink)  # Let training initialize from model zoo
# cfg.MODEL.WEIGHTS = ""

# number of images per batch
cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.SOLVER.IMS_PER_BATCH = 16

# learning rate
# cfg.SOLVER.BASE_LR = 0.0000025  # pick a good LR
if(parser.parse_args().learning_rate == -1):
  cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
else:
  cfg.SOLVER.BASE_LR = parser.parse_args().learning_rate


# cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
# cfg.SOLVER.BASE_LR = 0.001  # pick a good LR

# max iterations
# cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
# cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
if(parser.parse_args().max_iter == -1):
  cfg.SOLVER.MAX_ITER = 10000
else:
  cfg.SOLVER.MAX_ITER = parser.parse_args().max_iter


# Minibatch size PER image - number of regions of interest (ROIs)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)

# Number of classes
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(SharkClassDictionary)  # only has one class (ballon)
cfg.MODEL.RETINANET.NUM_CLASSES = len(SharkClassDictionary)  # only has one class (ballon)

# directories (?)
# Clear output directory

def CreateOutputFolder(counter):
  # Convert counter to a string
  ctrString = str(counter)
  
  # Append 0s
  if(counter < 100):
    ctrString = "0" + ctrString
    if(counter < 10):
      ctrString = "0" + ctrString

  # Create the folder and path names
  foldername = "output"+ctrString
  path = baseOutputDirectory + foldername

  # If it exists, recurse
  if(os.path.isdir(path)):
    nextNumber = counter + 1
    CreateOutputFolder(nextNumber)
  else:
    # Create the directory
    os.makedirs(path, exist_ok=True)
    cfg.OUTPUT_DIR = path

# Create a folder output0 etc.
CreateOutputFolder(0)

# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# filesInOutput = os.listdir(cfg.OUTPUT_DIR)
# for file in filesInOutput:
  # if(file != ".ipynb_checkpoints"):
    # os.remove(cfg.OUTPUT_DIR+"/"+file)

# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Get the default trainer:
## 1) Create model, optimiser, scheduler, dataloader from the given config
## 2) Load a checkpoint or cfg.MODEL.WEIGHTS if it exists
## 3) Register a few common hooks (?)
## This simplifies the standard model training workflow, so you don't have to write boilerplate code
# trainer = DefaultTrainer(cfg)
trainer = Trainer(cfg)

# data_loader = build_detection_train_loader(cfg, mapper=mapper)
# trainer.build_train_loader = data_loader

import datetime
dateTime = datetime.datetime.now()
dateTime = str(dateTime)

print("Model being used: ",modelLink)
print("Learning rate: ",cfg.SOLVER.BASE_LR)
print("Max iterations: ",cfg.SOLVER.MAX_ITER)
print("Number of classes: ",cfg.MODEL.RETINANET.NUM_CLASSES)

OutputString = "\nDate time: \t"    + dateTime \
             + "\n________________________________________________________" \
             + "\n\Model being used: \t" + modelLink \
             + "\nLearning rate: \t"     + str(cfg.SOLVER.BASE_LR) \
             + "\nMax iterations: \t"    + str(cfg.SOLVER.MAX_ITER) \
             + "\nNumber of classes: \t" + str(cfg.MODEL.RETINANET.NUM_CLASSES)

text_file = open(cfg.OUTPUT_DIR+"/parameters-information.txt", "w")
text_file.write(OutputString)
text_file.close()
# torch.save(OutputString,cfg.OUTPUT_DIR+"/parameters-information.txt")


# If true, and the last checkpoint exists, resume from it
# If false, load a model specified by the config
trainer.resume_or_load(resume=False)
trainer.train()

# Commented out IPython magic to ensure Python compatibility.
# Look at training curves in tensorboard:
# !kill 1825
# %load_ext tensorboard
# %tensorboard --logdir output

"""# Inference and Evaluation"""

# Inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.2
# cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.25
cfg.DATASETS.TEST = ("shark_val", )
# cfg.DATASETS.TEST = ("shark_train", )
# Create a simple end-to-end predictor with the given config
## This predictor takes care of model loading and input preprocessing for you
predictor = DefaultPredictor(cfg)

# Visualise:
# from detectron2.utils.visualizer import ColorMode
# dataset_dicts = getSharkTrainDicts()
dataset_dicts = getSharkValDicts() #getSharkDicts("/content/drive/My Drive/sharkdata/all_data/val")
# for dictionary in random.sample(dataset_dicts, 12):
for dictionary in random.sample(dataset_dicts, 12):
  im = cv2.imread(dictionary["file_name"])
  outputs = predictor(im)
  v = Visualizer(im[:, :, ::-1],
                  metadata=shark_metadata, 
                  scale=0.1,
                #  instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
  )

  classID = ((dictionary["annotations"])[0])["category_id"]
  sharkID = ClassList[classID]

  instances = outputs["instances"]
  classes = instances.get("pred_classes")
  sharkIDs = []
  for c in classes:
    sharkIDs.append(ClassList[c])
  # print(sharkIDs)
  scoresRaw = instances.get("scores")
  scores = []
  for s in scoresRaw:
    # sStr = str(s.item())
    # sStr = sStr[:4]
    s = s.item()
    s = round(s,2)
    scores.append(s)
  out = dict(zip(sharkIDs,scores))
  print(out)

  highestScoringClass = ""
  highestScore = 0.0
  for s in out:
    floatS = float(out[s])
    if(floatS > highestScore): 
      highestScore = floatS
      highestScoringClass = out

  if(sharkID in out):
    if(highestScoringClass == sharkID):
      print("Correct prediction, and highest predicted: ", sharkID, out[sharkID])
    else:
      print("Correct prediction: ", sharkID, out[sharkID])
  else:
    print("No prediction: ", sharkID, "0.00")

  v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  

  img = v.get_image()[:, :, ::-1]
  # cv2_imshow(img)
  # os.makedirs(baseDirectory + "outputs/", exist_ok=True)
  filename = baseDirectory + "outputs/" + sharkID + "_" + dictionary["file_name"]
  cv2.imwrite(filename, img)

# AP
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("shark_val", cfg, False, output_dir="./output/")
# evaluator = COCOEvaluator("shark_train", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "shark_val")
# val_loader = build_detection_test_loader(cfg, "shark_train")
inference_on_dataset(trainer.model, val_loader, evaluator)
# another equivalent way is to use trainer.test




# Open the file
text_file = open("jobName", "r")
# Read it in
jobName = text_file.read()
# Extract the jobname
jobName = jobName[20:len(jobName)-1]
# Create the file name
filename = "slurm-"+jobName+".out"
# Copy the file
shutil.copy("/mnt/storage/home/ja16475/sharks/detectron2/"+filename, cfg.OUTPUT_DIR+"/"+filename)
# Delete the original 
os.remove("/mnt/storage/home/ja16475/sharks/detectron2/"+filename)