print("Importing...")

import torch, torchvision

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import random
import math
import os
import json
import shutil

from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog

import argparse
import datetime
from collections import OrderedDict

from termcolor import colored, cprint

import datetime
import logging

from detectron2.utils.events import EventWriter
from detectron2.utils.events import get_event_storage

from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

from detectron2.structures import Instances

import evaluate
import mappers
import config
import getters
import writers
import train

print("Imports done")

# Get the datetime for logging purposes
dateTime = datetime.datetime.now()
dateTime = str(dateTime)


#-----------------------------------------------------#
#                      Parsing
#-----------------------------------------------------#
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
parser.add_argument(
  "-id",
  "--jobid",
  default=-1,
  type=int,
  help="The Slurm JOB ID - Not set by the user"
)
parser.add_argument(
  "-d",
  "--dataset",
  default="s",
  type=str,
  help="The dataset being used"
)
parser.add_argument(
  "-a",
  "--accuracy",
  default=0,
  type=int,
  help="Whether to track accuracy or not during training (this is *very* intensive)"
)
parser.add_argument(
  "-r",
  "--resume",
  default=-1,
  type=int,
  help="JobID to resume from"
)


dataset_used = ""
if(parser.parse_args().dataset == "s"):
  dataset_used = "small"
  print("Dataset being used is the small dataset")
elif(parser.parse_args().dataset == "l"):
  dataset_used = "large"
  print("Dataset being used is the large dataset")
else:
  raise ValueError("Dataset arg provided \""+parser.parse_args().dataset+"\" is invalid")

#-----------------------------------------------------#
#                     Directories
#-----------------------------------------------------#

trainDirectory      = ""
valDirectory        = ""
imageDirectory      = ""
sourceJsonDirectory = ""
baseDirectory       = ""
baseOutputDirectory = ""

if(dataset_used == "small"):
  trainDirectory      = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/small_set/photos/train/"
  valDirectory        = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/small_set/photos/val/"
  imageDirectory      = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/small_set/photos/images/"
  sourceJsonDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/small_set/photos/data.json"
  baseDirectory       = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/small_set/photos/"
  baseOutputDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/outputs/small/"
if(dataset_used == "large"):
  trainDirectory      = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/large_set/train/"
  valDirectory        = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/large_set/val/"
  imageDirectory      = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/large_set/images/"
  sourceJsonDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/large_set/data.json"
  baseDirectory       = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/large_set/"
  baseOutputDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/outputs/large/"


#-----------------------------------------------------#
#                  Handle ResumeID
#-----------------------------------------------------#
actualJobID = parser.parse_args().jobid
resumeID = parser.parse_args().resume #3380515
if(resumeID == -1):
  print("Training new model: ",actualJobID)
else:
  print("Resuming training from: ",resumeID)

#-----------------------------------------------------#
#                   Get Dicts
#-----------------------------------------------------#

myDictGetters = getters.DictionaryGetters(baseDirectory=baseDirectory,
                                          trainDirectory=trainDirectory,
                                          valDirectory=valDirectory)

SharkClassDictionary = myDictGetters.getSharkClassDictionary()
ClassList = myDictGetters.getClassList()


#-----------------------------------------------------#
#                  Register to Catalogs
#-----------------------------------------------------#
for d in ["train", "val"]:
  # Register shark_train and shark_val
  DatasetCatalog.register("shark_" + d, lambda d=d: myDictGetters.getSharkDicts(d))
  MetadataCatalog.get("shark_" + d).set(thing_classes=ClassList)

shark_metadata = MetadataCatalog.get("shark_train")




#-----------------------------------------------------#
#                Load the Config
#-----------------------------------------------------#
directory_to_load_from = "???"

modelLink = ""
modelOutputFolderName = ""
if(parser.parse_args().model == 0):
  modelLink = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"
  modelOutputFolderName = "retinanet_R_50_FPN_1x"
elif(parser.parse_args().model == 1):
  modelLink = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
  modelOutputFolderName = "retinanet_R_50_FPN_3x"
elif(parser.parse_args().model == 2):
  modelLink = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
  modelOutputFolderName = "retinanet_R_101_FPN_3x"
elif(parser.parse_args().model == 3):
  modelLink = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"
  modelOutputFolderName = "faster_rcnn_R_50_C4_1x"
elif(parser.parse_args().model == 4):
  modelLink = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
  modelOutputFolderName = "mask_rcnn_R_50_FPN_3x"
else:
  modelLink = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"
  modelOutputFolderName = "retinanet_R_50_FPN_1x"

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


# Load the CFG file
loadedCfg = torch.load(directory_to_load_from+"/cfg.yaml")
# Load in the weights
loadedCfg.MODEL.WEIGHTS = directory_to_load_from+"/model_final.pth"
# Change the output dir?
# loadedCfg.OUTPUT_DIR = "/content/outputs"

# Creat an output dir
jbName = str(parser.parse_args().jobid)
foldername = "output_"+jbName
path = baseOutputDirectory + modelOutputFolderName + "/" + foldername
os.makedirs(path, exist_ok=True)
loadedCfg.OUTPUT_DIR = path

# Build the model (this DOES NOT build in the weights)
loadedModel = build_model(loadedCfg)

# Load in the weights with the checkpointer
DetectionCheckpointer(loadedModel).load(directory_to_load_from+"/model_final.pth")


# helpful print/ wrinting function:
def PrintAndWriteToParams(stringToPrintWrite,writeType="w"):
  text_file = open(loadedCfg.OUTPUT_DIR+"/parameters-information.txt",writeType)
  text_file.write(stringToPrintWrite)
  text_file.close()

  print(stringToPrintWrite)

# Print / log cfg
jbName = str(parser.parse_args().jobid)
OutputString = "\nDate time: \t"    + dateTime \
             + "\nJobname: \t" + jbName \
             + "\nOutputting to: \t" + str(loadedCfg.OUTPUT_DIR) \
             + "\nLoading from: \t" + str(directory_to_load_from) \
             + "\n________________________________________________________" \
             + "\nModel being used: \t" + modelLink \
             + "\nModel index: \t\t" + str(parser.parse_args().model) \
             + "\nLearning rate: \t\t"     + str(loadedCfg.SOLVER.BASE_LR) \
             + "\nIteration at which LR starts to decrease by "+str(loadedCfg.SOLVER.GAMMA)+": "+str(loadedCfg.SOLVER.STEPS) \
             + "\nMax iterations: \t"    + str(loadedCfg.SOLVER.MAX_ITER) \
             + "\nImages per batch: \t"     + str(loadedCfg.SOLVER.IMS_PER_BATCH) \
             + "\nNumber of classes: \t" + str(loadedCfg.MODEL.RETINANET.NUM_CLASSES) \
             + "\n________________________________________________________" \
             + "\n"

PrintAndWriteToParams(OutputString)

#-----------------------------------------------------#
#                      Evaluate
#-----------------------------------------------------#
myEvaluator = evaluate.MyEvaluator(loadedCfg,loadedModel,dataset_used,myDictGetters)

# COCO Results
appendString = "\n________________________________________________________" \
              + "\nEvaluating the performance on TEST dataset" \
              + "\n"
PrintAndWriteToParams(appendString,"a+")

cocoResults = myEvaluator.EvaluateTestCOCO()

appendString = "\n________________________________________________________" \
              + "\nEvaluating the performance on TRAIN dataset" \
              + "\n"
PrintAndWriteToParams(appendString,"a+")

cocoResults = myEvaluator.EvaluateTrainCOCO()

appendString = "\n________________________________________________________" \
              + "\nEvaluating the performance on TEST dataset" \
              + "\n"
PrintAndWriteToParams(appendString,"a+")


## Top K
KAccDict = OrderedDict()
for i in range(1,11,2):
  accResult = myEvaluator.EvaluateTestTopKAccuracy(i)

appendString = "\n________________________________________________________" \
              + "\nEvaluating the performance on TRAIN dataset" \
              + "\n"
PrintAndWriteToParams(appendString,"a+")

KAccDict = OrderedDict()
for i in range(1,11,2):
  accResult = myEvaluator.EvaluateTrainTopKAccuracy(i)


#-----------------------------------------------------#
#                Visualise Predictions
#-----------------------------------------------------#
# test_dataset_dicts = myDictGetters.getSharkTrainDicts()
# evaluate.visualisePredictedExamples(myDictGetters,cfg,predictor,shark_metadata,15)


#-----------------------------------------------------#
#             FINALLY: Move the Slurm File
#-----------------------------------------------------#
# jobName = str(parser.parse_args().jobid)
jobName = str(actualJobID)
# Create the file name
filename = "slurm-"+jobName+".out"
print("Moving ",filename)
# Copy the file
shutil.copy("/mnt/storage/home/ja16475/sharks/detectron2/"+filename, cfg.OUTPUT_DIR+"/"+filename)
# Delete the original 
os.remove("/mnt/storage/home/ja16475/sharks/detectron2/"+filename)