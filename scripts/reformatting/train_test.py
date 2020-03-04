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
#                 Create the config
#-----------------------------------------------------#
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

cfg = config.CreateCfg(parser=parser.parse_args(),
                dataset_used=dataset_used,
                numClasses=len(SharkClassDictionary),
                baseOutputDir=baseOutputDirectory,
                modelLink=modelLink,
                modelOutputFolderName=modelOutputFolderName,
                jobIDOverride=resumeID)
                # if no resumeID is entered/parsed, it will be -1 and do nothing

#-----------------------------------------------------#
#              Create the Trainer
#-----------------------------------------------------#
# Create and evaluator to be used in training
# evaluator = evaluate.MyEvaluator(trainer.model,cfg,dataset_used)

if(dataset_used == "small"):
  trainer = train.SmallSetTrainer(cfg,parser.parse_args(),myDictGetters,dataset_used)
if(dataset_used == "large"):
  trainer = train.LargeSetTrainer(cfg,parser.parse_args(),myDictGetters,dataset_used)


# helpful print/ wrinting function:
def PrintAndWriteToParams(stringToPrintWrite,writeType="w"):
  text_file = open(cfg.OUTPUT_DIR+"/parameters-information.txt",writeType)
  text_file.write(stringToPrintWrite)
  text_file.close()

  print(stringToPrintWrite)

# Print / log cfg
jbName = str(parser.parse_args().jobid)
OutputString = "\nDate time: \t"    + dateTime \
             + "\nJobname: \t" + jbName \
             + "\nOutputting to: \t" + str(cfg.OUTPUT_DIR) \
             + "\n________________________________________________________" \
             + "\nModel being used: \t" + modelLink \
             + "\nModel index: \t\t" + str(parser.parse_args().model) \
             + "\nLearning rate: \t\t"     + str(cfg.SOLVER.BASE_LR) \
             + "\nIteration at which LR starts to decrease by "+str(cfg.SOLVER.GAMMA)+": "+str(cfg.SOLVER.STEPS) \
             + "\nMax iterations: \t"    + str(cfg.SOLVER.MAX_ITER) \
             + "\nImages per batch: \t"     + str(cfg.SOLVER.IMS_PER_BATCH) \
             + "\nNumber of classes: \t" + str(cfg.MODEL.RETINANET.NUM_CLASSES) \
             + "\n________________________________________________________" \
             + "\n"

PrintAndWriteToParams(OutputString)

#-----------------------------------------------------#
#                      Train
#-----------------------------------------------------#
# If true, and the last checkpoint exists, resume from it
# If false, load a model specified by the config
trainer.resume_or_load(resume=False)
trainer.train()


#-----------------------------------------------------#
#                      Evaluate
#-----------------------------------------------------#
# Load the final model outputted in training
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

# Attempt to create a predictor
# This may fail if training fails, in which case we move the slurm to the output folder
# and raise an error
try:
  predictor = DefaultPredictor(cfg)
except AssertionError:
  print("Checkpoint not found, model not found")
  ### Move the Slurm file ###
  # Get the jobname
  jobName = str(parser.parse_args().jobid)
  print("Moving ",jobName)
  # Create the file name
  filename = "slurm-"+jobName+".out"
  # Copy the file
  shutil.copy("/mnt/storage/home/ja16475/sharks/detectron2/"+filename, cfg.OUTPUT_DIR+"/"+filename)
  # Delete the original 
  os.remove("/mnt/storage/home/ja16475/sharks/detectron2/"+filename)
  raise AssertionError("model_final.pth not found! It's likely that training somehow failed.")


# Create an evaluation dictionary which we store as a file at the end of evaluation
evaluationDict = OrderedDict()

# Add parameters to the running evaluationDict
parameterDict = OrderedDict()
parameterDict["jobid"] = parser.parse_args().jobid
parameterDict["output_directory"] = cfg.OUTPUT_DIR
parameterDict["model"] = modelOutputFolderName
parameterDict["model_index"] = parser.parse_args().model
parameterDict["lr"] = cfg.SOLVER.BASE_LR
parameterDict["max_iter"] = cfg.SOLVER.MAX_ITER
parameterDict["batch_size_per_image"] = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
parameterDict["images_per_batch"] = cfg.SOLVER.IMS_PER_BATCH
parameterDict["num_classes"] = cfg.MODEL.RETINANET.NUM_CLASSES
parameterDict["transforms"] = "not implemented"
evaluationDict["params"] = parameterDict

# Do coco evaluation
myEvaluator = evaluate.MyEvaluator(cfg,trainer.model,dataset_used,myDictGetters)
cocoValResults = myEvaluator.EvaluateTestCOCO()
evaluationDict["coco_val"] = cocoValResults

cocoTrainResults = myEvaluator.EvaluateTrainCOCO()
evaluationDict["coco_train"] = cocoTrainResults


# Do Top K Test Accuracy
KAccDict = OrderedDict()
for i in range(1,11,2):
  accResult = myEvaluator.EvaluateTestTopKAccuracy(i)
  k = accResult["k"]
  key = "top_"+str(k)+"_acc"
  KAccDict[key] = accResult

evaluationDict["acc"] = KAccDict
torch.save(evaluationDict,cfg.OUTPUT_DIR+"/evaluationDictionary.pt")


# Create the string we're going to add to the text_file
appendString = "\n________________________________________________________" \
              + "\nEvaluating the performance on training dataset" \
              + "\n"
PrintAndWriteToParams(appendString,"a+")


# Do Top K Train Accuracy
for i in range(1,11,2):
  myEvaluator.EvaluateTrainTopKAccuracy(i)

# Append to file
appendString = "\n________________________________________________________" \
              + "\n"
PrintAndWriteToParams(appendString,"a+")


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