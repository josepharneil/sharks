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
# from detectron2.engine import DefaultPredictor
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
import ModelPaths
import MyPredictor
import MyDirectoryHandler
# import RetinaNetOHEM, DropoutRetinaNet

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

print("Imports done")

print("Torch cuda version:",torch.version.cuda)

# Get the datetime for logging purposes
dateTime = datetime.datetime.now()
dateTime = str(dateTime)


#-----------------------------------------------------#
#                      Parsing
#-----------------------------------------------------#
parser = argparse.ArgumentParser(description="Train over small shark set",formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

parser.add_argument("-lr","--learning-rate",default=-1,type=float,help="Base learning rate used by the model")
parser.add_argument("-m","--model",default=-1,type=int,help="Model used")
parser.add_argument("-i","--max-iter",default=-1,type=int,help="Max number of iterations in training")
parser.add_argument("-id","--jobid",default=-1,type=int,help="The Slurm JOB ID - Not set by the user")
parser.add_argument("-d","--dataset",default=2,type=int,help="The dataset being used")
parser.add_argument("-a","--accuracy",default=0,type=int,help="Whether to track accuracy or not during training (this is *very* intensive)")
parser.add_argument("-r","--resume",default=-1,type=int,help="JobID to resume from")
parser.add_argument("-b","--batch-size",default=0,type=int,help="Batch size")
parser.add_argument("-t","--threshold",default=800,type=int,help="Image thresholder")
parser.add_argument("-tt","--test-time",default=1,type=int,help="Test-time or not")
parser.add_argument("-c","--curriculum",default=0,type=int,help="0 no curriculum, 1 curriculum, 2 anti-curriculum")
parser.add_argument("-cid","--curriculum_id",default=0,type=int,help="curriculum id, similar to resume")
parser.add_argument("-cr","--crop",default=1,type=int,help="Crop to bbox or not")
parser.add_argument("-op","--optimiser",default=0,type=int,help="Which optimiser: 0 sgd; 1 adam; 2 adagrad")
# parser.add_argument("-xv","--cross-val",default=-1,type=int,help="Which cross-val set")

if( (parser.parse_args().resume not in [-1,0]) and (parser.parse_args().curriculum in [1,2] )):
  raise NotImplementedError("I haven't bothered to implement resuming and curriculum learning yet \n\t- need to be able to set appropriate dataloader depending on iter")


dataset_used = ""
# if(parser.parse_args().dataset == "s"):
if(parser.parse_args().dataset == 0):
  dataset_used = "small"
  print("Dataset being used is the small dataset")
# elif(parser.parse_args().dataset == "l"):
elif(parser.parse_args().dataset == 1):
  dataset_used = "large"
  print("Dataset being used is the large dataset")
# elif(parser.parse_args().dataset == "f"):
elif(parser.parse_args().dataset == 2):
  dataset_used = "full"
  print("Dataset being used is the full dataset")
elif(parser.parse_args().dataset == 20):
  dataset_used = "split0"
  print("Dataset being used is the full dataset, split 0")
elif(parser.parse_args().dataset == 21):
  dataset_used = "split1"
  print("Dataset being used is the full dataset, split 1")
elif(parser.parse_args().dataset == 22):
  dataset_used = "split2"
  print("Dataset being used is the full dataset, split 2")
elif(parser.parse_args().dataset == 23):
  dataset_used = "split3"
  print("Dataset being used is the full dataset, split 3")
elif(parser.parse_args().dataset == 24):
  dataset_used = "split4"
  print("Dataset being used is the full dataset, split 4")
# elif(parser.parse_args().dataset == "c"):
elif(parser.parse_args().dataset == 3):
  dataset_used = "comparison"
  print("Dataset being used is the comparison dataset")
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

baseOutputDirectory, baseDirectory, trainDirectory, valDirectory, imageDirectory, sourceJsonDirectory = MyDirectoryHandler.GetDatasetDirectories(dataset_used)

# if(dataset_used == "small"):
#   baseOutputDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/outputs/small/"
#   baseDirectory       = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/small_set/photos/"
#   trainDirectory      = baseDirectory + "train/"
#   valDirectory        = baseDirectory + "val/"
#   imageDirectory      = baseDirectory + "images/"
#   sourceJsonDirectory = baseDirectory + "data.json"
# elif(dataset_used == "large"):
#   baseOutputDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/outputs/large/"
#   baseDirectory       = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/large_set/"
#   trainDirectory      = baseDirectory + "train/"
#   valDirectory        = baseDirectory + "val/"
#   imageDirectory      = baseDirectory + "images/"
#   sourceJsonDirectory = baseDirectory + "data.json"
# elif(dataset_used == "full"):
#   baseOutputDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/outputs/full/"
#   baseDirectory       = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/full_set/"
#   trainDirectory      = baseDirectory + "train/"
#   valDirectory        = baseDirectory + "val/"
#   imageDirectory      = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/large_set/" + "images/"
#   sourceJsonDirectory = baseDirectory + "data.json"
# elif(dataset_used == "split0"):
#   baseOutputDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/outputs/full/"
#   baseDirectory       = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/full_set/splits/0_split/"
#   trainDirectory      = baseDirectory + "train/"
#   valDirectory        = baseDirectory + "val/"
#   imageDirectory      = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/large_set/" + "images/"
#   sourceJsonDirectory = baseDirectory + "data.json"
# elif(dataset_used == "split1"):
#   baseOutputDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/outputs/full/"
#   baseDirectory       = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/full_set/splits/2_split/"
#   trainDirectory      = baseDirectory + "train/"
#   valDirectory        = baseDirectory + "val/"
#   imageDirectory      = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/large_set/" + "images/"
#   sourceJsonDirectory = baseDirectory + "data.json"
# elif(dataset_used == "split2"):
#   baseOutputDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/outputs/full/"
#   baseDirectory       = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/full_set/splits/3_split/"
#   trainDirectory      = baseDirectory + "train/"
#   valDirectory        = baseDirectory + "val/"
#   imageDirectory      = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/large_set/" + "images/"
#   sourceJsonDirectory = baseDirectory + "data.json"
# elif(dataset_used == "split3"):
#   baseOutputDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/outputs/full/"
#   baseDirectory       = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/full_set/splits/3_split/"
#   trainDirectory      = baseDirectory + "train/"
#   valDirectory        = baseDirectory + "val/"
#   imageDirectory      = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/large_set/" + "images/"
#   sourceJsonDirectory = baseDirectory + "data.json"
# elif(dataset_used == "split4"):
#   baseOutputDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/outputs/full/"
#   baseDirectory       = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/full_set/splits/4_split/"
#   trainDirectory      = baseDirectory + "train/"
#   valDirectory        = baseDirectory + "val/"
#   imageDirectory      = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/large_set/" + "images/"
#   sourceJsonDirectory = baseDirectory + "data.json"
# elif(dataset_used == "comparison"):
#   baseOutputDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/outputs/comparison/"
#   baseDirectory       = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/comparison_set/"
#   trainDirectory      = baseDirectory + "train/"
#   valDirectory        = baseDirectory + "val/"
#   imageDirectory      = baseDirectory + "images/"
#   sourceJsonDirectory = baseDirectory + "data.json"
# else:
#   raise ValueError("dataset_used doesn't exist:"+dataset_used)

#-----------------------------------------------------#
#                  Handle ResumeID
#-----------------------------------------------------#
actualJobID = parser.parse_args().jobid
resumeID = parser.parse_args().resume #3380515
if(resumeID == -1 or resumeID == 0):
  print("Training new model: ",actualJobID)
else:
  print("Resuming training from: ",resumeID)


#-----------------------------------------------------#
#               Handle Model Parsed Arg
#-----------------------------------------------------#
modelLink,modelOutputFolderName,meta_arch_override = ModelPaths.GetModelLinks(parser.parse_args().model)

'''
modelLink = ""
modelOutputFolderName = ""
if(parser.parse_args().model == 0):
  modelLink = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"
  modelOutputFolderName = "retinanet_R_50_FPN_1x"
  meta_arch_override = None
elif(parser.parse_args().model == 1):
  modelLink = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
  modelOutputFolderName = "retinanet_R_50_FPN_3x"
  meta_arch_override = None
elif(parser.parse_args().model == 2):
  modelLink = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
  modelOutputFolderName = "retinanet_R_101_FPN_3x"
  meta_arch_override = None
elif(parser.parse_args().model == 3):
  modelLink = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
  modelOutputFolderName = "faster_rcnn_X_101_32x8d_FPN_3x"
  meta_arch_override = None
elif(parser.parse_args().model == 4):
  modelLink = "VGG19_BN"
  modelOutputFolderName = "VGG19_BN"
  meta_arch_override = None
elif(parser.parse_args().model == 5):
  modelLink = "YOLOV3"
  modelOutputFolderName = "YOLOV3"
  meta_arch_override = None
elif(parser.parse_args().model == 6):
  modelLink = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
  modelOutputFolderName = "retinanet_R_101_FPN_3x_OHEM"
  meta_arch_override = "RetinaNetOHEM"
elif(parser.parse_args().model == 7):
  modelLink = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
  modelOutputFolderName = "retinanet_R_101_FPN_3x_DROPOUT"
  meta_arch_override = "DropoutRetinaNet"
else:
  raise ValueError("No such model index:", parser.parse_args().model)
'''

#-----------------------------------------------------#
#              Handle Curriculum override
#-----------------------------------------------------#
# If we're not doing curriculum, no override
isShuffleData = True
if(parser.parse_args().curriculum == 0):
  curriculum_override = None
# if we ARE doing curriculum
else:
  if(parser.parse_args().curriculum_id == 0):
    raise ValueError("If we're doing curriculum learning, we need to set a curriculum ID as well")
  else:
    currID = parser.parse_args().curriculum_id

  # If curriculum
  if(parser.parse_args().curriculum == 1):
    curriculum_override = baseOutputDirectory + modelOutputFolderName + "/" + "output_"+str(currID)+"/sharkTrainDicts-curriculum.pt"
    isShuffleData = False
  # If anti-curriculum
  if(parser.parse_args().curriculum == 2):
    curriculum_override = baseOutputDirectory + modelOutputFolderName + "/" + "output_"+str(currID)+"/sharkTrainDicts-anti-curriculum.pt"
    isShuffleData = False
  # If NO curriculum
  else:
    curriculum_override = None


#-----------------------------------------------------#
#               Get Dicts/ Dict Getters
#-----------------------------------------------------#
myDictGetters = getters.DictionaryGetters(baseDirectory=baseDirectory,
                                          trainDirectory=trainDirectory,
                                          valDirectory=valDirectory,
                                          curriculum_override=curriculum_override)

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

cfg = config.CreateCfg(parser=parser.parse_args(),
                dataset_used=dataset_used,
                numClasses=len(SharkClassDictionary),
                baseOutputDir=baseOutputDirectory,
                modelLink=modelLink,
                modelOutputFolderName=modelOutputFolderName,
                jobIDOverride=resumeID,
                meta_arch_override=meta_arch_override,
                optim=parser.parse_args().optimiser)
                # if no resumeID is entered/parsed, it will be -1 and do nothing

#-----------------------------------------------------#
#              Create the Trainer
#-----------------------------------------------------#
# Create and evaluator to be used in training
# evaluator = evaluate.MyEvaluator(trainer.model,cfg,dataset_used)

# if(dataset_used == "small"):
#   trainer = train.SmallSetTrainer(cfg,parser.parse_args(),myDictGetters,dataset_used,threshold_dimension,is_test_time_mapping)
# if(dataset_used == "large"):
#   trainer = train.LargeSetTrainer(cfg,parser.parse_args(),myDictGetters,dataset_used,threshold_dimension,is_test_time_mapping)
# if(dataset_used == "full"):
#   trainer = train.FullSetTrainer(cfg,parser.parse_args(),myDictGetters,dataset_used,threshold_dimension,is_test_time_mapping)
# if(dataset_used == "comparison"):
#   trainer = train.ComparisonSetTrainer(cfg,parser.parse_args(),myDictGetters,dataset_used,threshold_dimension,is_test_time_mapping)

threshold_dimension = parser.parse_args().threshold
is_test_time_mapping = True if (parser.parse_args().test_time == 1) else False
if(is_test_time_mapping): 
  print("is test time mapping")
else:
  print("is not test time mapping")

if(isShuffleData): print("Shuffling data")
else:              print("Not shuffling data")

is_crop_to_bbox = True if (parser.parse_args().crop == 1) else False
if(is_crop_to_bbox): print("Cropping to bbox")
else:                print("Not cropping to bbox")

trainer = train.My_Trainer(cfg,parser.parse_args(),myDictGetters,dataset_used,threshold_dimension,is_test_time_mapping,modelLink,isShuffleData,is_crop_to_bbox)

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

PrintAndWriteToParams("\nTransforms: crop-to-bbox, rescale, brightness etc., AFFINE: shear (-8,8) in each axis + rotations (-30,30) + scale (0.9)\n","a+")
# PrintAndWriteToParams("\nNo test time data augmentation (except crop-to-bbox, and rescaling) \n","a+")

PrintAndWriteToParams("Model type: "+str(type(trainer.model)),"a+")

#-----------------------------------------------------#
#                      Train
#-----------------------------------------------------#
# If true, and the last checkpoint exists, resume from it
# If false, load a model specified by the config


# Don't resume
if(resumeID == -1 or resumeID == 0):
  trainer.resume_or_load(resume=False)  
  print("Don't resume")
# Do resume
else:
  trainer.resume_or_load(resume=True)
  print("Resume")
  print(trainer.checkpointer.save_dir)

trainer.train()


#-----------------------------------------------------#
#                      Evaluate
#-----------------------------------------------------#
# Load the final model outputted in training
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

# Attempt to create a predictor
# This may fail if training fails, in which case we move the slurm to the output folder
# and raise an error
# if(modelLink not in ["VGG19_BN","YOLOV3"] ):
try:
  # predictor = DefaultPredictor(cfg)
  predictor = MyPredictor.MyPredictor(cfg)
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

is_crop_to_bbox = True if (parser.parse_args().crop == 1) else False
if(is_crop_to_bbox): print("Cropping to bbox")
else:                print("Not cropping to bbox")
# Create evaluator object
myEvaluator = evaluate.MyEvaluator(cfg,trainer.model,dataset_used,myDictGetters,threshold_dimension,is_test_time_mapping,is_crop_to_bbox)

# coco
# cocoValResults = myEvaluator.EvaluateTestCOCO()
# evaluationDict["coco_val"] = cocoValResults

# cocoTrainResults = myEvaluator.EvaluateTrainCOCO()
# evaluationDict["coco_train"] = cocoTrainResults

# AP
pathToAPFolder = cfg.OUTPUT_DIR + "/AP_Evaluation"
os.makedirs(pathToAPFolder, exist_ok=True)

def EvaluateAPatIOU(IOU):
  # Get the interp data at IOU
  interp_data_XX = myEvaluator.EvaluateTestAP(IOU)
  stringIOU = str(int(IOU * 100))
  # Save the dictionary for future plotting
  torch.save(interp_data_XX,pathToAPFolder+"/interp_data_"+stringIOU+".pt")
  # Get the AP
  AP_at_XX_For_Class = evaluate.GetAPForClass(interp_data_XX,"overall")
  # Print and append to file
  appendString = "Overall AP at IOU "+stringIOU+": " + str(AP_at_XX_For_Class) + "\n"
  PrintAndWriteToParams(appendString,"a+")

  return AP_at_XX_For_Class

if(not modelLink == "VGG19_BN"):
  AP_At_50 = EvaluateAPatIOU(0.5)


# Do Top K Test Accuracy
KAccDict = OrderedDict()
for i in range(1,11,2):
  accResult = myEvaluator.EvaluateTestTopKAccuracy(i)
  k = accResult["k"]
  key = "top_"+str(k)+"_acc"
  KAccDict[key] = accResult
  if(i == 1):
    testAccAt1 = str(accResult["accuracy"]) + "%"

evaluationDict["acc"] = KAccDict
torch.save(evaluationDict,cfg.OUTPUT_DIR+"/evaluationDictionary.pt")


# Create the string we're going to add to the text_file
appendString = "\n________________________________________________________" \
              + "\nEvaluating the performance on training dataset" \
              + "\n"
PrintAndWriteToParams(appendString,"a+")


# Do Top K Train Accuracy
for i in range(1,11,2):
  train_accResult = myEvaluator.EvaluateTrainTopKAccuracy(i)
  if(i == 1):
    trainAccAt1 = str(train_accResult["accuracy"]) + "%"

# Append to file
appendString = "\n________________________________________________________" \
              + "\n"
PrintAndWriteToParams(appendString,"a+")

if(not modelLink == "VGG19_BN"):
  appendString = "\nRESULT: ap50, train, test : " + str(round(AP_At_50,3)) + ", " + trainAccAt1 + ", " + testAccAt1 + "\n"
  PrintAndWriteToParams(appendString,"a+")

  # appendString = "\nLaTeX friendly: ap50, train, test : " + str(round(AP_At_50,3)) + " & " + trainAccAt1 + "\\% & " + testAccAt1 + "\n"
  # PrintAndWriteToParams(appendString,"a+")
else:
  appendString = "\nRESULT: train, test : " + trainAccAt1 + ", " + testAccAt1 + "\n"
  PrintAndWriteToParams(appendString,"a+")

PrintAndWriteToParams("Finished evaluation!\n","a+")


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