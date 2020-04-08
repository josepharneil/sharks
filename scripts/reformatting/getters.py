import torch

class DictionaryGetters():
  def __init__(self,baseDirectory,trainDirectory,valDirectory,curriculum_override=None):
    self.baseDirectory  = baseDirectory
    self.trainDirectory = trainDirectory
    self.valDirectory   = valDirectory
    self.curriculum_override = curriculum_override

  # Used in registering datasets to catalogs
  def getSharkDicts(self,trainVal):
    if(trainVal == "train"):
      return self.getSharkTrainDicts()
    if(trainVal == "val"):
      return self.getSharkValDicts()

  # Called by getSharkDicts
  def getSharkTrainDicts(self):
    # If we're not overriding the standard train dict
    if(self.curriculum_override == None):
      return torch.load(self.trainDirectory+"sharkTrainDicts.pt")
    # If we're using curriculum learning
    else:
      return torch.load(self.curriculum_override)

  # Called by getSharkDicts
  def getSharkValDicts(self):
    return torch.load(self.valDirectory+"sharkValDicts.pt")

  def getSharkClassDictionary(self):
    return torch.load(self.baseDirectory+"SharkClassDictionary.pt")

  def getClassList(self):
    return torch.load(self.baseDirectory+"ClassList.pt")