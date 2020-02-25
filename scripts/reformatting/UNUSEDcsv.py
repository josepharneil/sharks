import csv
with open(cfg.OUTPUT_DIR+"/output.csv", "w") as outputCSV:
  writer = csv.writer(outputCSV)
  for key,value in evaluationDict.items():
    writer.writerow([key,value])


def AppendToCSV():
  paramKeys = list(evaluationDict["params"].keys())
  paramVals = list(evaluationDict["params"].values())

  cocoKeys = list(evaluationDict["coco"].keys())
  cocoKeys = cocoKeys[:len(cocoKeys)-1]
  cocoVals = list(evaluationDict["coco"].values())
  cocoVals = cocoVals[:len(cocoVals)-1]

  accVals = []
  for acc in list((evaluationDict["acc"].values())):
    accVals.append(acc["accuracy"])

  accKeys = list(evaluationDict["acc"].keys())

  perClass = evaluationDict["coco"].pop("PerClassAP")

  perClassKeyList = []
  perClassValList = []
  for key,value in perClass.items():
    perClassKeyList.append(key)
    perClassValList.append(value)

  cocoKeys = cocoKeys + perClassKeyList
  cocoVals = cocoVals + perClassValList

  resultKeys = paramKeys + accKeys + cocoKeys
  resultVals = paramVals + accVals + cocoVals

  # Creation of the csv and adding new keys will be done in a different script
  # Append new values
  csvFilename = "result_"+dataset_used+".csv"
  with open(csvFilename,"a+") as outputCSV:
    writer = csv.writer(outputCSV)
    # writer.writerow(resultKeys)
    writer.writerow(resultVals)

############### END CSV Nightmare ###############

AppendToCSV()
