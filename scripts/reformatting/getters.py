import torch

class DictionaryGetters():
  def __init__(self,baseDirectory,trainDirectory,valDirectory):
    self.baseDirectory  = baseDirectory
    self.trainDirectory = trainDirectory
    self.valDirectory   = valDirectory

  # Used in registering datasets to catalogs
  def getSharkDicts(self,trainVal):
    if(trainVal == "train"):
      return self.getSharkTrainDicts()
    if(trainVal == "val"):
      return self.getSharkValDicts()

  # Called by getSharkDicts
  def getSharkTrainDicts(self):
    return torch.load(self.trainDirectory+"sharkTrainDicts.pt")

  # Called by getSharkDicts
  def getSharkValDicts(self):
    return torch.load(self.valDirectory+"sharkValDicts.pt")

  def getSharkClassDictionary(self):
    return torch.load(self.baseDirectory+"SharkClassDictionary.pt")

  def getClassList(self):
    return torch.load(self.baseDirectory+"ClassList.pt")