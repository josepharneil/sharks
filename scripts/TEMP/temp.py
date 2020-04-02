# import json

# # json_file = os.path.join(dataDirectory, "data.json")

# # Open the file, referring to it as f
# with open(json_file) as f:
#   # Get the image annotations out using json to read the file containin the JSON object
# img_annotations = json.load(f)

import torch

trainDicts = torch.load("TEMP/sharkTrainDicts.pt")
valDicts = torch.load("TEMP/sharkValDicts.pt")

for trainDict in trainDicts:
  file_name = trainDict["file_name"]
  file_name = "/content/drive/My Drive/sharkdata/all_data/images/" + file_name.split("images/")[1]
  trainDict["file_name"] = file_name

for valDict in valDicts:
  file_name = valDict["file_name"]
  file_name = "/content/drive/My Drive/sharkdata/all_data/images/" + file_name.split("images/")[1]
  valDict["file_name"] = file_name


torch.save(trainDicts,"TEMP/sharkTrainDicts_modified.pt")
torch.save(valDicts,"TEMP/sharkValDicts_modified.pt")