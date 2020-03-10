import torch
import numpy as np

from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
import copy

from PIL import Image
from torchvision.transforms.functional import _get_inverse_affine_matrix
from detectron2.data.transforms.transform import Transform

def apply_affine(affMat,x,y):
  # print(x,y)
  a,b,c,d,e,f = affMat
  newX = a*x + b*y + c
  newY = d*x + e*y + f
  # print(a,b,c,d,e,f)
  # print(newX)
  # print(newY)
  return newX,newY

class RandomAffineTransform(Transform):
  def __init__(self, imageSize, angle=0, translate=(0,0), scale=0.9):
    center = (imageSize[0] * 0.5 + 0.5, imageSize[1] * 0.5 + 0.5)
    shear = (np.random.uniform(-8,8),np.random.uniform(-8,8))
    angle = np.random.uniform(-30,30)

    self.invAffMat = _get_inverse_affine_matrix(center=center, angle=angle, translate=translate, scale=scale, shear=shear)

    invAffM = np.mat([ [self.invAffMat[0],self.invAffMat[1],self.invAffMat[2] ],
                       [self.invAffMat[3],self.invAffMat[4],self.invAffMat[5] ],
                       [0        ,0        ,1                  ] ])
    affMat = np.linalg.inv(invAffM)
    self.affMat = affMat.item(0),affMat.item(1),affMat.item(2),affMat.item(3),affMat.item(4),affMat.item(5)

  def apply_image(self, image):
    PILImage = Image.fromarray(image)
    output_size = PILImage.size
    AFFImage = PILImage.transform(size=output_size,method=Image.AFFINE,data=self.invAffMat,resample=0)
    return np.array(AFFImage)

  def apply_coords(self, coords):
    # NOTE: THIS JUST CONVERTS COORDS, CHANGES BOXES TO PARALELOGRAMS, WHICH MAY BREAK THINGS??
    # print(type(coords))
    # print("coords: ",coords)
    # xmin,ymin,xmax,ymax = coords
    # print(": ",coords)
    p0,p1,p2,p3 = coords[0],coords[1],coords[2],coords[3]
    # p0,p1,p2,p3=(xmin,ymin),(xmax,ymax),(xmax,ymin),(xmin,ymax)

    # print(p0,p1,p2,p3)

    # Apply the affine equation
    p0 = apply_affine(self.affMat,p0[0],p0[1])
    p1 = apply_affine(self.affMat,p1[0],p1[1])
    p2 = apply_affine(self.affMat,p2[0],p2[1])
    p3 = apply_affine(self.affMat,p3[0],p3[1])

    # Get the box min maxes
    # xs = [p0[0],p1[0],p2[0],p3[0]]
    # ys = [p0[1],p1[1],p2[1],p3[1]]
    # # print(xs,ys)
    # xmin = int(round(min(xs)))
    # ymin = int(round(min(ys)))
    # xmax = int(round(max(xs)))
    # ymax = int(round(max(ys)))

    # Set the new bbox in the dictionary
    # bbox = [xmin,ymin,xmax,ymax]
    return np.array((p0,p1,p2,p3))


def train_mapper(dataset_dict,dataset_used):
  # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
  # Create a copy of the dataset dict
  dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below


  ##### Image Transformations #####
  # Read in the image
  image = utils.read_image(dataset_dict["file_name"], format="BGR")
  # fileName = dataset_dict["file_name"]

  ## Crop to bounding box ##
  # Get the bounding box
  bbox = ((dataset_dict["annotations"])[0])["bbox"]
  xmin,ymin,xmax,ymax = bbox
  w = xmax-xmin
  h = ymax-ymin

  # Nudge the crop to be slightly outside of the bounding box
  nudgedXMin = xmin-15
  nudgedYMin = ymin-15
  nudgedW = w+50
  nudgedH = h+50

  # If the bounding boxes go outside of the image dimensions, fix this
  imageHeight = image.shape[0]
  imageWidth  = image.shape[1]
  if(nudgedXMin < 0): nudgedXMin = 0
  if(nudgedYMin < 0): nudgedYMin = 0
  if(nudgedXMin+nudgedW >= imageWidth):  nudgedW = imageWidth-1
  if(nudgedYMin+nudgedH >= imageHeight): nudgedH = imageHeight-1

  # Apply the crop
  cropT = T.CropTransform(nudgedXMin,nudgedYMin,nudgedW,nudgedH)
  image = cropT.apply_image(image)

  # Apply the crop to the bbox as well
  # THIS IS HANDLED IN annotations_to_instances, so long as this is appended to the list of transforms

  dataset_dict["height"] = nudgedH
  dataset_dict["width"]  = nudgedW
  
  # Add to the list of transforms
  transforms = T.TransformList([cropT])

  ## Scale the image size ##
  # thresholdDimension = 1000
  # if(dataset_used == "large"):
    # thresholdDimension = 500
  thresholdDimension = 800

  # Downscale only at this threshold
  if(nudgedH > thresholdDimension or nudgedW > thresholdDimension):
    myNewH = 0
    myNewW = 0
    # Scale the longest dimension to 1333, the shorter to 800
    if(nudgedH > nudgedW): 
      myNewH = thresholdDimension
      ratio = nudgedH/float(myNewH)
      myNewW = nudgedW/float(ratio)
      myNewW = int(round(myNewW))
      # myNewW = 800
    else:
      # myNewH = 800
      myNewW = thresholdDimension
      ratio = nudgedW/float(myNewW)
      myNewH = nudgedH/float(ratio)
      myNewH = int(round(myNewH))

    # Apply the scaling transform
    scaleT = T.ScaleTransform(h=nudgedH,w=nudgedW,new_h=myNewW,new_w=myNewH,interp="nearest") 
    image = scaleT.apply_image(image.copy())

    # Apply the scaling to the bbox
    # THIS IS HANDLED IN annotations_to_instances, so long as this is appended to the list of transforms

    # Add this to the list of transforms
    transforms = transforms + scaleT

    # Set the dimensions
    dataset_dict["height"] = myNewH
    dataset_dict["width"]  = myNewW
  
  ## Apply a random flip ##
  image, tfms = T.apply_transform_gens([T.RandomFlip()], image)
  transforms = transforms + tfms

  # Apply Other Transforms ##
  image, tfms = T.apply_transform_gens([T.RandomBrightness(0.4,1.6),T.RandomContrast(0.4,1.6),T.RandomSaturation(0.5,1.5),T.RandomLighting(1.2)], image)
  transforms = transforms + tfms

  ## Apply random affine (actually just a shear) ##
  # Pass in the image size
  PILImage = Image.fromarray(image)
  RandAffT = RandomAffineTransform(PILImage.size)
  # Apply affine to image
  image = RandAffT.apply_image(image.copy())
  # Append to transforms
  transforms = transforms + RandAffT

  ##### END Image Transformations #####

  # Keep these in for now I suppose
  if(image.shape[0] == 0): 
    raise ValueError("image shape[0] is 0!: ",print(image.shape),dataset_dict["file_name"])
  if(image.shape[1] == 0): 
    raise ValueError("image shape[1] is 0!: ",print(image.shape),dataset_dict["file_name"])

  # Set the image in the dictionary
  dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))


  # Do remainder of dictionary
  classID = ((dataset_dict["annotations"])[0])["category_id"]
  dataset_dict["classID"] = classID

  annos = \
  [
    utils.transform_instance_annotations(obj, transforms, image.shape[:2])
    for obj in dataset_dict.pop("annotations")
    if obj.get("iscrowd", 0) == 0
  ]

  # transformNames = [transforms.__name__ for x in transforms]
  # transformNames = ", ".join(transformNames)

  instances = utils.annotations_to_instances(annos, image.shape[:2])
  dataset_dict["instances"] = utils.filter_empty_instances(instances)

  dataset_dict["transforms"] = transforms

  return dataset_dict

def test_mapper(dataset_dict,dataset_used):

  ## temp ##
  return train_mapper(dataset_dict,dataset_used)
  ########
   
  # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
  # Create a copy of the dataset dict
  dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below


  ##### Image Transformations #####
  # Read in the image
  image = utils.read_image(dataset_dict["file_name"], format="BGR")
  # fileName = dataset_dict["file_name"]

  ## Crop to bounding box ##
  # Get the bounding box
  bbox = ((dataset_dict["annotations"])[0])["bbox"]
  xmin,ymin,xmax,ymax = bbox
  w = xmax-xmin
  h = ymax-ymin

  # Nudge the crop to be slightly outside of the bounding box
  nudgedXMin = xmin-15
  nudgedYMin = ymin-15
  nudgedW = w+50
  nudgedH = h+50

  # If the bounding boxes go outside of the image dimensions, fix this
  imageHeight = image.shape[0]
  imageWidth  = image.shape[1]
  if(nudgedXMin < 0): nudgedXMin = 0
  if(nudgedYMin < 0): nudgedYMin = 0
  if(nudgedXMin+nudgedW >= imageWidth):  nudgedW = imageWidth-1
  if(nudgedYMin+nudgedH >= imageHeight): nudgedH = imageHeight-1

  # Apply the crop
  cropT = T.CropTransform(nudgedXMin,nudgedYMin,nudgedW,nudgedH)
  image = cropT.apply_image(image)

  # Apply the crop to the bbox as well
  
  dataset_dict["height"] = nudgedH
  dataset_dict["width"]  = nudgedW
  
  # Add to the list of transforms
  transforms = T.TransformList([cropT])

  ## Scale the image size ##
  thresholdDimension = 1000
  # if(dataset_used == "large"):
    # thresholdDimension = 500
  thresholdDimension = 800

  # Downscale only at this threshold
  if(nudgedH > thresholdDimension or nudgedW > thresholdDimension):
    myNewH = 0
    myNewW = 0
    # Scale the longest dimension to 1333, the shorter to 800
    if(nudgedH > nudgedW): 
      myNewH = thresholdDimension
      ratio = nudgedH/float(myNewH)
      myNewW = nudgedW/float(ratio)
      myNewW = int(round(myNewW))
      # myNewW = 800
    else:
      # myNewH = 800
      myNewW = thresholdDimension
      ratio = nudgedW/float(myNewW)
      myNewH = nudgedH/float(ratio)
      myNewH = int(round(myNewH))

    # Apply the scaling transform
    scaleT = T.ScaleTransform(h=nudgedH,w=nudgedW,new_h=myNewW,new_w=myNewH,interp="nearest") 
    image = scaleT.apply_image(image.copy())

    # Add this to the list of transforms
    transforms = transforms + scaleT

    # Set the dimensions
    dataset_dict["height"] = myNewH
    dataset_dict["width"]  = myNewW
  
  ## Apply a random flip ##
  # image, tfms = T.apply_transform_gens([T.RandomFlip()], image)
  # transforms = transforms + tfms

  # Apply Other Transforms ##
  # image, tfms = T.apply_transform_gens([T.RandomBrightness(0.4,1.6),T.RandomContrast(0.4,1.6),T.RandomSaturation(0.5,1.5),T.RandomLighting(1.2)], image)
  # transforms = transforms + tfms

  ## Apply random affine (actually just a shear) ##
  # Pass in the image size
  # PILImage = Image.fromarray(image)
  # RandAffT = RandomAffineTransform(PILImage.size)
  # Apply affine to image
  # image = RandAffT.apply_image(image.copy())
  # Append to transforms
  # transforms = transforms + RandAffT

  ##### END Image Transformations #####

  # Keep these in for now I suppose
  if(image.shape[0] == 0): 
    raise ValueError("image shape[0] is 0!: ",print(image.shape),dataset_dict["file_name"])
  if(image.shape[1] == 0): 
    raise ValueError("image shape[1] is 0!: ",print(image.shape),dataset_dict["file_name"])

  # Set the image in the dictionary
  dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))


  # Do remainder of dictionary
  classID = ((dataset_dict["annotations"])[0])["category_id"]
  dataset_dict["classID"] = classID

  annos = \
  [
    utils.transform_instance_annotations(obj, transforms, image.shape[:2])
    for obj in dataset_dict.pop("annotations")
    if obj.get("iscrowd", 0) == 0
  ]

  # transformNames = [transforms.__name__ for x in transforms]
  # transformNames = ", ".join(transformNames)

  instances = utils.annotations_to_instances(annos, image.shape[:2])
  dataset_dict["instances"] = utils.filter_empty_instances(instances)

  return dataset_dict


def small_train_mapper(dataset_dict):
  return train_mapper(dataset_dict,"small")

def small_test_mapper(dataset_dict):
  return test_mapper(dataset_dict,"small")

def large_train_mapper(dataset_dict):
  return train_mapper(dataset_dict,"large")

def large_test_mapper(dataset_dict):
  return test_mapper(dataset_dict,"large")

def full_train_mapper(dataset_dict):
  return train_mapper(dataset_dict,"full")

def full_test_mapper(dataset_dict):
  return test_mapper(dataset_dict,"full")