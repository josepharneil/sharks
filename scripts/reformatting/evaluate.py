import torch
from collections import OrderedDict

from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation import DatasetEvaluator
from detectron2.data import build_detection_test_loader

import time
from contextlib import contextmanager
from detectron2.evaluation.evaluator import DatasetEvaluators
from termcolor import colored, cprint
import copy
import math


import mappers


#-----------------------------------------------------#
#               Define the TopKAcc Class
#-----------------------------------------------------#

class TopKAccuracy(DatasetEvaluator):
  def __init__(self, getter, dataset_used, cfg=None, k=5, output_images=False):
    self.k = k
    self.perClassDict = OrderedDict()
    self.ClassList = getter.getClassList()
    self.dataset_used = dataset_used
    self.output_images = output_images
    self.bboxSaveDict = {}
    self.cfg = cfg

  def reset(self):
    self.numberCorrect = 0
    self.totalNumber   = 0
    self.perClassDict = OrderedDict()
  # I think this is a single batch, with the inputted images and outputted results
  def process(self, inputs, outputs):
    for input,output in zip(inputs,outputs):
      # Increment the total number no matter what
      self.totalNumber = self.totalNumber + 1

      # Get the true class ID
      classID = input["classID"]
      trueSharkID = self.ClassList[classID]

      # Increment the total number in this class's dict entry
      # (numCorrect,totalNum,list of incorrect filenames)
      # Add the filename, it will be removed if this is counts as correct
      currentFilename = input["file_name"]
      # If this sharkID already exists
      if(trueSharkID in self.perClassDict.keys()):
        newNumCorrect = self.perClassDict[trueSharkID][0]
        newTotalNum = self.perClassDict[trueSharkID][1]
        if(self.dataset_used == "small"):
          # Get the current list
          newList = (self.perClassDict[trueSharkID][2])
          # Append
          newList.append(currentFilename)
        else: # when using "large"
          newList = []
        self.perClassDict[trueSharkID] = ( newNumCorrect, newTotalNum + 1, newList )
      else:
        self.perClassDict[trueSharkID] = (0,1,[currentFilename])

      # Get the instances object from the outputs
      instances = output["instances"]
      
      # Get the predicted classes for this image
      classes = instances.get("pred_classes")
      # Convert classes to more useful sharkIDs
      predictedSharkIDs = []
      for c in classes:
        predictedSharkIDs.append(self.ClassList[c])

      # Get the list of scores for each prediction
      scores = instances.get("scores")
      scores = scores.cpu()
      scores = scores.numpy()

      bboxes = instances.get("pred_boxes")
      bboxes = [box.cpu().numpy() for box in bboxes]
      
      # If there are no predicted scores for his input, skip iteration of the loop
      if(len(scores) == 0): continue

      # Zip up the predicted shark IDs and scores into a dictionary
      # sharkIDScoreDict = dict(zip(predictedSharkIDs,scores))
      sharkIDScoreList = list(zip(predictedSharkIDs,scores,bboxes))
      # Sort it into a list of descending order, in order of the value (the score)
      # sortedSharkIDScoreList = sorted(sharkIDScoreDict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
      sortedSharkIDScoreList = sorted(sharkIDScoreList, key = lambda kv:(kv[1], kv[0]), reverse=True)

      # sortedSharkIDScoreList is a list of tuples: [(sharkID,score,bbox),...]
      # sortedSharkIDScoreList[0] gives you the highest scoring tuple
      # (sortedSharkIDScoreList[0])[0] gives you the sharkID for the 0th tuple

      # Get the top K shark IDs
      # topKPredictedIDs = []
      for i in range(0,self.k):
        # If the list is shorter than the number of k's we want to look at, 
        # then break, no need to continue as there are no more predictions to consider
        if(i >= len(sortedSharkIDScoreList)): break
        # Extract ith tuple
        currentTuple = sortedSharkIDScoreList[i]
        # Get the shark ID
        currentPredID = currentTuple[0]
        currentScore = currentTuple[1]
        # currentBbox = currentTuple[2]
        if(self.output_images):
          # Only save the highest predicted
          if(i == 0):
            try: 
              transforms = output["transforms"]
            except:
              transforms = "broke"
              # print(output.keys())

            saveTuple = (currentTuple[0],currentTuple[1],currentTuple[2],transforms)
            self.bboxSaveDict[currentFilename] = saveTuple

        # Append this to the top K predictions
        # topKPredictedIDs.append(currentPredID)

        # We increase the rank of the correct prediction for each equivalence we find
        # So if there are many predictions with the same score to the correct prediction
        # we are "lowering" its rank to not consider it as much 
        # (where rank 0 is the highest)
        rank = -1

        # # If we're dealing with the true one
        # if(currentPredID == trueSharkID):
        #   # Go through all pairs
        #   for idx,scoreSharkIDPair in enumerate(sortedSharkIDScoreList):
        #     # If there is an equivalent score
        #     if(scoreSharkIDPair[1] == currentScore):
        #       rank = idx
        #       break

        # If the current predictedID we are considering is the trueSharkID
        if(currentPredID == trueSharkID):
          # Compare the correct prediction's score to all other scores
          for idx,scoreSharkIDPair in enumerate(sortedSharkIDScoreList):
            # If there is an equivalence in score
            if(scoreSharkIDPair[1] == currentScore):
              # If the rank hasn't been initialised, 
              # set it to the lowest index of this score
              if(rank == -1): 
                rank = idx + 1
              # If the rank has been set, increment
              else:
                # Increment the rank
                # Note, this will occur at least once as we compare it to itself
                rank = rank + 1
          # If the rank has exceed the number k we wanted to look at, don't count it
          if(rank <= self.k):
            # We increment, and then we don't care about the rest of the k's
            self.numberCorrect = self.numberCorrect + 1

            # Increment the correct of this class
            # Remove the filename

            # if(len(newList) > 1)
            newNumCorrect = self.perClassDict[trueSharkID][0]
            newTotalNum = self.perClassDict[trueSharkID][1]
            newList = (self.perClassDict[trueSharkID][2])
            if(self.dataset_used == "small"):
              newList.remove(currentFilename)

            self.perClassDict[currentPredID] = ( newNumCorrect+1, newTotalNum, newList )
            break

  # Return a dictionary of the final result
  def evaluate(self):

    if(self.output_images):
      torch.save(self.bboxSaveDict, self.cfg.OUTPUT_DIR+"/bboxSaveDict.pt")

    # Sort the dictionary by proportion correct
    self.perClassDict = OrderedDict(sorted(self.perClassDict.items(), key=lambda t: (t[1])[0]/(t[1])[1]))

    accuracy = float(self.numberCorrect) / float(self.totalNumber)
    # {"total_num": self.totalNumber, "num_correct": self.numberCorrect, "accuracy": accuracy, "k": self.k, "perClass": self.perClassDict}
    result = OrderedDict()
    # result["total_num"] = self.totalNumber
    # result["num_correct"] = self.numberCorrect
    # result["accuracy"] = accuracy
    # result["k"] = self.k
    # result["perClass"] = self.perClassDict

    result["topKAcc"] = {"total_num": self.totalNumber, "num_correct": self.numberCorrect, "accuracy": accuracy, "k": self.k, "perClass": self.perClassDict}

    return result

    # return {"total_num": self.totalNumber, "num_correct": self.numberCorrect, "accuracy": accuracy, "k": self.k, "perClass": self.perClassDict}




@contextmanager
def inference_context(model): #this is used in inference_on_dataset
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

def inference_on_dataset(model, data_loader, evaluator): # modified version of inference_on_dataset
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    print(colored("Calculating inference...","green"))
    # num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    # logger = logging.getLogger(__name__)
    # logger.info("Start inference on {} images".format(len(data_loader)))

    # total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    # num_warmup = min(5, total - 1)
    # start_time = time.perf_counter()
    # total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            # if idx == num_warmup:
                # start_time = time.perf_counter()
                # total_compute_time = 0

            # start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            # total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            # iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            # seconds_per_img = total_compute_time / iters_after_start
            # if idx >= num_warmup * 2 or seconds_per_img > 5:
                # total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                # eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                # log_every_n_seconds(
                #     logging.INFO,
                #     "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                #         idx + 1, total, seconds_per_img, str(eta)
                #     ),
                #     n=5,
                # )

    # Measure the time only for this worker (before the synchronization barrier)
    # total_time = time.perf_counter() - start_time
    # total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    # logger.info(
    #     "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
    #         total_time_str, total_time / (total - num_warmup), num_devices
    #     )
    # )
    # total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    # logger.info(
    #     "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
    #         total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
    #     )
    # )
    print(colored("Done calculating","green"))

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results





class MyEvaluator():
  def __init__(self,cfg,model,dataset_used,getter):
    self.cfg = cfg
    self.dataset_used = dataset_used
    self.getter = getter
    self.model = model

  def EvaluateTopKAccuracy(self,testOrTrain,numK,isReturn=False):
    # Decide if we're evaluating test or train set
    if(testOrTrain == "test"):
      datasetName = "shark_val"
    elif(testOrTrain == "train"):
      datasetName = "shark_train"
    else:
      raise ValueError("Evaluate Top K Accuracy: Dataset inputted doesn't exist!"+testOrTrain)

    # Set up the val_loader with the appropriate mapper
    if(self.dataset_used == "small"):
      val_loader = build_detection_test_loader(self.cfg, datasetName, mapper=mappers.small_test_mapper)
    elif(self.dataset_used == "large"):
      val_loader = build_detection_test_loader(self.cfg, datasetName, mapper=mappers.large_test_mapper)
    else:
      raise ValueError("Evaluate Top K Accuracy: Dataset inputted doesn't exist!"+self.dataset_used)

    # Create evaluator object
    topKEvaluator = TopKAccuracy(getter=self.getter,dataset_used=self.dataset_used,k=numK)
    # Get the accuracy results
    # val_loader = build_detection_test_loader(cfg, "shark_val", mapper=test_mapper)
    # Note to self: self.model used to be trainer.model
    accuracy_results = inference_on_dataset(self.model, val_loader, topKEvaluator)

    if(isReturn):
      return accuracy_results["topKAcc"]
    else:
      accuracy_results = accuracy_results["topKAcc"]
      # Extract results
      total_num   = str(accuracy_results["total_num"])
      num_correct = str(accuracy_results["num_correct"])
      top_k_acc   = str(round((accuracy_results["accuracy"]*100),2)) + "%"
      k           = str(accuracy_results["k"])
      # Per class is an ordered dictionary of classIDs mapping to triples of the form 
      # (numCorrect,totalNum,list of incorrectly classified filenames)
      # OD( classID: (numCorr,totNum,list),...  )
      perClass    = accuracy_results["perClass"]

      perClassFiles = ""
      perClassString = " Class  || prop  | numCorrect | totalNum\n"
      for key,value in perClass.items():
        # 6 chars long
        currClass = key

        # 3 chars long
        currCorrect = value[0]
        currCorrectStr = str(currCorrect)
        if(currCorrect < 100):
          currCorrectStr = "0"+currCorrectStr
          if(currCorrect < 10): 
            currCorrectStr = "0"+currCorrectStr

        # 3 chars long
        currTotal = value[1]
        currTotalStr = str(currTotal)
        if(currTotal < 100):
          currTotalStr = "0"+currTotalStr
          if(currTotal < 10): 
            currTotalStr = "0"+currTotalStr

        # 4 chars long
        currPropCorrect = str(round(((float(currCorrect)/float(currTotal))*100),2))
        # If the string is of form 12.3, make it 12.30
        if(len(currPropCorrect) == 4):
          currPropCorrect = currPropCorrect + '0'

        # Per class string
        perClassString = perClassString + (" "+currClass+" || "+currPropCorrect+" |    "+currCorrectStr+"     |   "+currTotalStr+"\n")

        # Class:filenames
        listOfFilenames = value[2]
        seperator = ', '
        perClassFiles = perClassFiles + (currClass+": [" + seperator.join(listOfFilenames) + "]\n")


      # Create the string we're going to add to the text_file
      appendString = "\n________________________________________________________" \
                  + "\nNumber correct: \t" + num_correct \
                  + "\nTotal Number: \t\t" + total_num \
                  + "\nTop " + str(k) + " Accuracy: \t" + top_k_acc \
                  + "\n"

      appendString = appendString + perClassString
      appendString = appendString + perClassFiles

      # Append to the file
      text_file = open(self.cfg.OUTPUT_DIR+"/parameters-information.txt", "a+")
      text_file.write(appendString)
      text_file.close()

      # Print the file
      print(  "\nNumber correct: \t" + num_correct \
            + "\nTotal Number: \t\t" + total_num \
            + "\nTop " + str(k) + " Accuracy: \t" + top_k_acc \
            + "\n\n")

      result = OrderedDict()
      result["accuracy"]    = round(accuracy_results["accuracy"]*100,2)
      result["num_correct"] = accuracy_results["num_correct"]
      result["total_num"]   = accuracy_results["total_num"]
      result["k"]           = accuracy_results["k"]
      return result

  def EvaluateTestTopKAccuracy(self,numK,isReturn=False):
    return self.EvaluateTopKAccuracy("test",numK,isReturn=False)

  def EvaluateTrainTopKAccuracy(self,numK,isReturn=False):
    return self.EvaluateTopKAccuracy("train",numK,isReturn=False)

  def EvaluateCOCO(self,dataset_to_eval):
    # The loader for the test data (applies various transformations if we so choose)
    # val_loader = build_detection_test_loader(self.cfg, "shark_val", mapper=test_mapper)
    # Set up the val_loader with the appropriate mapper
    if(self.dataset_used == "small"):
      val_loader = build_detection_test_loader(self.cfg, dataset_to_eval, mapper=mappers.small_test_mapper)
    elif(self.dataset_used == "large"):
      val_loader = build_detection_test_loader(self.cfg, dataset_to_eval, mapper=mappers.large_test_mapper)
    else:
      raise ValueError("Evaluate Top K Accuracy: Dataset inputted doesn't exist!",self.dataset_used)

    print("Evaluating using COCO Metrics: ", dataset_to_eval)
  
    # Get the coco evaluator
    cocoEvaluator = COCOEvaluator(dataset_to_eval, self.cfg, False, output_dir=self.cfg.OUTPUT_DIR+"/")

    # Run the model on the data_loader and evaluate the metrics evaluator
    # Also benchmarks the inference speed of model.forward accurately
    cocoOutput = inference_on_dataset(self.model, val_loader, cocoEvaluator)

    # "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"]
    cocoBbox = cocoOutput["bbox"]
    mAP   = round(cocoBbox["AP"],2)
    mAP50 = round(cocoBbox["AP50"],2)
    mAP75 = round(cocoBbox["AP75"],2)
    mAPs  = round(cocoBbox["APs"],2)
    mAPm  = round(cocoBbox["APm"],2)
    mAPl  = round(cocoBbox["APl"],2)
    
    copycocoB = copy.deepcopy(cocoBbox)
    copycocoB.pop("AP")
    copycocoB.pop("AP50")
    copycocoB.pop("AP75")
    copycocoB.pop("APs")
    copycocoB.pop("APm")
    copycocoB.pop("APl")

    numAPClasses = 0
    averageScore = 0
    for APClass in copycocoB:
      numAPClasses = numAPClasses + 1
      if(not math.isnan(copycocoB[APClass])):
        averageScore = averageScore + copycocoB[APClass]

    if(numAPClasses != 0):
      averageScore = float(averageScore) / float(numAPClasses)
    else:
      averageScore = float('nan')
    # averageScore = str(averageScore)

    # print(cocoBbox["AP"])
    # another equivalent way is to use trainer.test

    # Create the string we're going to add to the text_file
    appendString = "\n________________________________________________________" \
                  + "\nEvaluating: "+ dataset_to_eval \
                  + "\nAverage Precision COCO: \t" + str(mAP) \
                  + "\nAverage Precision 50  : \t" + str(mAP50) \
                  + "\nAverage Precision 75  : \t" + str(mAP75) \
                  + "\nAverage Precision Small : \t" + str(mAPs) \
                  + "\nAverage Precision Medium: \t" + str(mAPm) \
                  + "\nAverage Precision Large : \t" + str(mAPl) \
                  + "\nMean Average Precision: \t" + str(averageScore) \
                  + "\n"

    # Append to the file
    text_file = open(self.cfg.OUTPUT_DIR+"/parameters-information.txt", "a+")
    text_file.write(appendString)
    text_file.close()

    resultDict = OrderedDict()
    resultDict["AP"]   = mAP
    resultDict["AP50"] = mAP50
    resultDict["AP75"] = mAP75
    resultDict["APs"]  = mAPs
    resultDict["APm"]  = mAPm
    resultDict["APl"]  = mAPl
    resultDict["ClassAveAP"] = averageScore
    resultDict["PerClassAP"] = copycocoB
    return resultDict


  def EvaluateTestCOCO(self):
    return self.EvaluateCOCO("shark_val")

  def EvaluateTrainCOCO(self):
    return self.EvaluateCOCO("shark_train")



import random
import cv2
import os
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances

# Visualise:
# from detectron2.utils.visualizer import ColorMode
def visualisePredictedExamples(getter, cfg, predictor, shark_metadata, num):
  dataset_dicts = getter.getSharkValDicts()
  ClassList = getter.getClassList()

  # dataset_dicts = getSharkTrainDicts()
  # dataset_dicts = getSharkValDicts()
  for dictionary in random.sample(dataset_dicts, num):
    im = cv2.imread(dictionary["file_name"])
    outputs = predictor(im)
    vis = Visualizer(im[:, :, ::-1],
                    metadata=shark_metadata, 
                    # scale=0.1,
                  #  instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )

    classID = ((dictionary["annotations"])[0])["category_id"]
    sharkID = ClassList[classID]

    instances = outputs["instances"]
    classes = instances.get("pred_classes")
    sharkIDs = []
    for c in classes:
      sharkIDs.append(ClassList[c])
    scoresRaw = instances.get("scores")
    scores = []
    for s in scoresRaw:
      s = s.item()
      s = round(s,2)
      scores.append(s)
    out = dict(zip(sharkIDs,scores))
    # print(out)

    highestScoringClass = ""
    highestScore = 0.0
    highestScoreIndex = -1
    for i,s in enumerate(out):
      floatS = float(out[s])
      if(floatS > highestScore): 
        highestScore = floatS
        highestScoringClass = out
        highestScoreIndex = i

    if(highestScoreIndex != -1):
      predboxes   = instances.get_fields()["pred_boxes"][highestScoreIndex]
      predscores  = instances.get_fields()["scores"][highestScoreIndex]
      predscores = torch.tensor([predscores.item()])
      predclasses = instances.get_fields()["pred_classes"][highestScoreIndex]
      predclasses = torch.tensor([predclasses.item()])
      myInst = Instances((im.shape[0],im.shape[1]))
      newFields = {"pred_boxes":predboxes,"scores":predscores,"pred_classes":predclasses}
      myInst.set("pred_boxes",predboxes)
      myInst.set("scores",predscores)
      myInst.set("pred_classes",predclasses)

    # if(sharkID in out):
    #   if(highestScoringClass == sharkID):
    #     print("Correct prediction, and highest predicted: ", sharkID, out[sharkID])
    #   else:
    #     print("Correct prediction: ", sharkID, out[sharkID])
    # else:
    #   print("No prediction: ", sharkID, "0.00")

    # If there is a highest prediction, then draw it
    if(highestScoreIndex != -1):
      v = vis.draw_instance_predictions(myInst.to("cpu"))
    else:
      v = vis.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Draw the groundtruth
    v = vis.draw_dataset_dict(dictionary)

    # Get the image
    img = v.get_image()[:, :, ::-1]

    # Create the predictions folder
    os.makedirs(cfg.OUTPUT_DIR + "/predictions", exist_ok=True)
    # Set up filename
    imageFilename = dictionary["image_id"] + "_" + sharkID + ".jpg"
    imageFilename = cfg.OUTPUT_DIR + "/predictions/" + imageFilename
    # Write the image
    cv2.imwrite(imageFilename,img)
    print("Saving image: ",imageFilename)

    # initialPath = os.getcwd()
    # os.makedirs(cfg.OUTPUT_DIR + "/predictions", exist_ok=True)
    # os.chdir(cfg.OUTPUT_DIR + "/predictions")
    # imageFilename = dictionary["file_name"] + "_" + sharkID + ".jpg"
    # cv2.imwrite(imageFilename, img)
    # os.chdir(initialPath)
    # filename = cfg.OUTPUT_DIR + "/predictions/" + dictionary["file_name"] + "_" + sharkID + ".jpg"
