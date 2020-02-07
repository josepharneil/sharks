import csv  
import os
import shutil
import argparse

parser = argparse.ArgumentParser(
    description="Create the CSV",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
  "-d",
  "--dataset",
  default="s",
  type=str,
  help="The dataset being used."
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



csvFilename = "result_"+dataset_used+".csv"

keys = ['jobid', 'output_directory', 'model', 'model_index', 'lr', 'max_iter', 'batch_size_per_image', 'num_classes', 'transforms', 'top_1_acc', 'top_3_acc', 'top_5_acc', 'top_7_acc', 'top_9_acc', 'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'ClassAveAP', 'AP-R-1008', 'AP-R-1005', 'AP-R-1007', 'AP-R-1014', 'AP-R-1004', 'AP-R-1053', 'AP-R-1039', 'AP-R-1018', 'AP-R-1029', 'AP-R-1038', 'AP-R-1012', 'AP-R-1045', 'AP-R-1028', 'AP-R-1046', 'AP-R-1047', 'AP-R-1022', 'AP-R-1021', 'AP-R-1056', 'AP-R-1058', 'AP-R-1137', 'AP-R-1064', 'AP-R-1070', 'AP-R-1020', 'AP-R-1002', 'AP-R-1066', 'AP-R-1010', 'AP-R-1067', 'AP-R-1001', 'AP-R-1209', 'AP-R-1230', 'AP-R-1305', 'AP-R-1096', 'AP-R-1068', 'AP-R-1304', 'AP-R-1158', 'AP-R-1303', 'AP-R-1184', 'AP-R-1294', 'AP-R-1289', 'AP-R-1327', 'AP-R-1049', 'AP-R-1188', 'AP-R-1182', 'AP-R-1257', 'AP-R-1156', 'AP-R-1278', 'AP-R-1333', 'AP-R-1210', 'AP-R-1099', 'AP-R-1270', 'AP-R-1292', 'AP-R-1109', 'AP-R-1190', 'AP-R-1194', 'AP-R-1275', 'AP-R-1241', 'AP-R-1195', 'AP-R-1203', 'AP-R-1329', 'AP-R-1279', 'AP-R-1136', 'AP-R-1341', 'AP-R-1035', 'AP-R-1166', 'AP-R-1273', 'AP-R-1027', 'AP-R-1215', 'AP-R-1343', 'AP-R-1342', 'AP-R-1059', 'AP-R-1115', 'AP-R-1155', 'AP-R-1293', 'AP-R-1296', 'AP-R-1213', 'AP-R-1344', 'AP-R-1207', 'AP-R-1345', 'AP-R-1161', 'AP-R-1117', 'AP-R-1291', 'AP-R-1153', 'AP-R-1316', 'AP-R-1272', 'AP-R-1337', 'AP-R-1308', 'AP-R-1350', 'AP-R-1348', 'AP-R-1355', 'AP-R-1358', 'AP-R-1347', 'AP-R-1060', 'AP-R-1362', 'AP-R-1361', 'AP-R-1321', 'AP-R-1364', 'AP-R-1299', 'AP-R-1368', 'AP-R-1371', 'AP-R-1248', 'AP-R-1375', 'AP-R-1377', 'AP-R-1378', 'AP-R-1376', 'AP-R-1380', 'AP-R-1383', 'AP-R-1352', 'AP-R-1236', 'AP-R-1385', 'AP-R-1284', 'AP-R-1317', 'AP-R-1178', 'AP-R-1353', 'AP-R-1386', 'AP-R-1369', 'AP-R-1265', 'AP-R-1382', 'AP-R-1208', 'AP-R-1387', 'AP-R-1334', 'AP-R-1390']

# If it doesn't already exist, create it
if(not os.path.exists(csvFilename)):
  with open(csvFilename,"w") as outputCSV:
    writer = csv.writer(outputCSV)
    writer.writerow(keys)
    # reader = csv.reader(outputCSV)
    # for row in reader:
      # row is a list of strings

print("Created ",csvFilename)

# If we want to add a new key
# keyName will just be whatever we want to call it
# previousKey is the key before it, so we will insert after this "previousKey"
def AddNewKey(keyName, previousKey, defaultValue):
  # Get the current list of keys from the csv file
  currentKeys = []
  with open(csvFilename,"r") as outputCSV:
    reader = csv.reader(outputCSV)
    for row in reader:
      currentKeys = row
      break
  
  # Get the previousKey's index
  prevIndex = -1
  if(previousKey in currentKeys):
    prevIndex = currentKeys.index(previousKey) + 1
  else:
    if(previousKey == ""):
      prevIndex = 0
    else:
      raise ValueError("previousKey \""+previousKey+"\" does not exist in the current list of keys")
  
  # Add the new key
  currentKeys.insert(prevIndex,keyName)

  # Now write it into the csv

  with open(csvFilename,"r") as original:
    with open("temp.csv","w") as temp:
      reader = csv.reader(original)
      writer = csv.writer(temp)
      for i,row in enumerate(reader):
        # If its the top row (the keys)
        if (i == 0):
          # Write our new keys in
          writer.writerow(currentKeys)
        # All other lines, just write in the values
        else:
          # List of strings
          currRow = row
          currRow.insert(prevIndex,defaultValue)
          writer.writerow(currRow)
  
  # Delete the old file
  os.remove(csvFilename)

  # Rename the new file
  os.rename("temp.csv",csvFilename)

# AddNewKey("test_test!","output_directory","default_val")