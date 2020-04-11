import torch
import numpy as np

from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.structures import Instances
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
  def __init__(self, imageSize, shear=(0,0), angle=0, translate=(0,0), scale=0.9):
    center = (imageSize[0] * 0.5 + 0.5, imageSize[1] * 0.5 + 0.5)
    # shear = (np.random.uniform(-8,8),np.random.uniform(-8,8))
    # angle = np.random.uniform(-30,30)

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


class My_Mapper():
  def __init__(self,dataset_used,threshold_dimension=800,is_test_time_mapping=False,modelLink="",is_crop_to_bbox=True):
    self.dataset_used = dataset_used
    self.threshold_dimension = threshold_dimension
    self.is_test_time_mapping = is_test_time_mapping
    self.modelLink = modelLink
    self.is_crop_to_bbox = is_crop_to_bbox

  def train_mapper(self,dataset_dict):#,dataset_used):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    # Create a copy of the dataset dict
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below


    ##### Image Transformations #####
    # Read in the image
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    # fileName = dataset_dict["file_name"]

    
    ## Crop to bounding box ##
    # Crop for all but comparison
    if(self.dataset_used != "comparison" and self.is_crop_to_bbox):
      # Get the bounding box
      bbox = ((dataset_dict["annotations"])[0])["bbox"]
      xmin,ymin,xmax,ymax = bbox
      w = xmax-xmin
      h = ymax-ymin

      # IsCropToBBox = True
      # if(IsCropToBBox):
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
      
      transforms = T.TransformList([cropT])
    # Comparison has bbox the size of the image, so dont bother cropping
    else:
      # scaled between 0.5 and 1; shifted up to 0.5 in each dimension
      # randomExtant = T.RandomExtent( (0.5,1),(0.5,0.5) )
      # transforms = T.TransformList([randomExtant])
      transforms = T.TransformList([])

    # Apply the crop to the bbox as well
    # THIS IS HANDLED IN annotations_to_instances, so long as this is appended to the list of transforms

    dataset_dict["height"] = image.shape[0]
    dataset_dict["width"]  = image.shape[1]
    
    # Add to the list of transforms
    
    # else:
    #   nudgedH = dataset_dict["height"]
    #   nudgedW = dataset_dict["width"]
      

    ## Scale the image size ##
    # thresholdDimension = 1000
    # if(dataset_used == "large"):
      # thresholdDimension = 500
    # thresholdDimension = 800
    # thresholdDimension = 600
    thresholdDimension = self.threshold_dimension

    currWidth  = dataset_dict["width"]
    currHeight = dataset_dict["height"]

    # NOTE: YOLO input size must be multiple of 32
    if(self.modelLink in ["VGG19_BN","YOLOV3"]):
      vgg_im_size = thresholdDimension
      # Apply the scaling transform
      scaleT = T.ScaleTransform(h=currHeight,w=currWidth,new_h=vgg_im_size,new_w=vgg_im_size,interp="nearest") 
      image = scaleT.apply_image(image.copy())

      # Apply the scaling to the bbox
      # THIS IS HANDLED IN annotations_to_instances, so long as this is appended to the list of transforms

      # Add this to the list of transforms
      transforms = transforms + scaleT

      # Set the dimensions
      dataset_dict["height"] = image.shape[0]
      dataset_dict["width"]  = image.shape[1]
    else:# Downscale only at this threshold
      if(currHeight > thresholdDimension or currWidth > thresholdDimension):
        myNewH = 0
        myNewW = 0
        # Scale the longest dimension to 1333, the shorter to 800
        if(currHeight > currWidth): 
          myNewH = thresholdDimension
          ratio = currHeight/float(myNewH)
          myNewW = currWidth/float(ratio)
          myNewW = int(round(myNewW))
          # myNewW = 800
        else:
          # myNewH = 800
          myNewW = thresholdDimension
          ratio = currWidth/float(myNewW)
          myNewH = currHeight/float(ratio)
          myNewH = int(round(myNewH))

        # Apply the scaling transform
        scaleT = T.ScaleTransform(h=currHeight,w=currWidth,new_h=myNewW,new_w=myNewH,interp="nearest") 
        image = scaleT.apply_image(image.copy())

        # Apply the scaling to the bbox
        # THIS IS HANDLED IN annotations_to_instances, so long as this is appended to the list of transforms

        # Add this to the list of transforms
        transforms = transforms + scaleT

        # Set the dimensions
        dataset_dict["height"] = image.shape[0]
        dataset_dict["width"]  = image.shape[1]
    
    ## Apply a random flip ##
    image, tfms = T.apply_transform_gens([T.RandomFlip()], image)
    transforms = transforms + tfms

    # Apply Other Transforms ##
    # Standard random image mods
    if(self.dataset_used != "comparison"):
      image, tfms = T.apply_transform_gens([T.RandomBrightness(0.4,1.6),T.RandomContrast(0.4,1.6),T.RandomSaturation(0.5,1.5),T.RandomLighting(1.2)], image)
    # More extreme for comparison set
    else:
      image, tfms = T.apply_transform_gens([T.RandomBrightness(0.2,1.8),T.RandomContrast(0.2,1.8),T.RandomSaturation(0.3,1.7),T.RandomLighting(1.5)], image)
    transforms = transforms + tfms

    ## Apply random affine (actually just a shear) ##
    # Pass in the image size
    PILImage = Image.fromarray(image)

    # Standard affine
    if(self.dataset_used != "comparison"):
      shear_range = 8
      angle_range = 30
      # rand_shear = (np.random.uniform(-shear_range,shear_range),np.random.uniform(-8,8))
      # rand_angle = np.random.uniform(-30,30)
    # More extreme random affine for comparison
    else:
      shear_range = 50
      angle_range = 30
      # rand_shear = (np.random.uniform(-30,30),np.random.uniform(-30,30))
      # rand_angle = np.random.uniform(-70,70)

    rand_shear = (np.random.uniform(-shear_range,shear_range),np.random.uniform(-shear_range,shear_range))
    rand_angle = np.random.uniform(-angle_range,angle_range)

    RandAffT = RandomAffineTransform(PILImage.size,shear=rand_shear,angle=rand_angle)
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

    # bboxes
    # if(self.dataset_used != "comparison"):
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
    # # no bboxes
    # else:
    #   instances = Instances(  (dataset_dict["height"],dataset_dict["width"])  )
    #   instances.gt_classes = torch.tensor([dataset_dict["classID"]])
    #   dataset_dict["instances"] = instances

    dataset_dict["transforms"] = transforms

    return dataset_dict

  def test_mapper(self,dataset_dict):#,dataset_used):
    # If we're mapping at test time
    if(self.is_test_time_mapping):
      return self.train_mapper(dataset_dict)



    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    # Create a copy of the dataset dict
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below


    ##### Image Transformations #####
    # Read in the image
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    # fileName = dataset_dict["file_name"]

    
    ## Crop to bounding box ##
    if(self.dataset_used != "comparison" and self.is_crop_to_bbox):
      # Get the bounding box
      bbox = ((dataset_dict["annotations"])[0])["bbox"]
      xmin,ymin,xmax,ymax = bbox
      w = xmax-xmin
      h = ymax-ymin

      # IsCropToBBox = True
      # if(IsCropToBBox):
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
      
      transforms = T.TransformList([cropT])
    # else:
      # nudgedH = dataset_dict["height"]
      # nudgedW = dataset_dict["width"]
    else:
      transforms = T.TransformList([])

      
    # Apply the crop to the bbox as well
    # THIS IS HANDLED IN annotations_to_instances, so long as this is appended to the list of transforms

    dataset_dict["height"] = image.shape[0]
    dataset_dict["width"]  = image.shape[1]
    
    # Add to the list of transforms
    
    # else:
    #   nudgedH = dataset_dict["height"]
    #   nudgedW = dataset_dict["width"]
      

    ## Scale the image size ##
    # thresholdDimension = 1000
    # if(dataset_used == "large"):
      # thresholdDimension = 500
    # thresholdDimension = 800
    # thresholdDimension = 600
    thresholdDimension = self.threshold_dimension

    currWidth  = dataset_dict["width"]
    currHeight = dataset_dict["height"]

    # the way ive done vgg and yolo means they need the same size images
    if(self.modelLink in ["VGG19_BN","YOLOV3"]):
      vgg_im_size = thresholdDimension
      # Apply the scaling transform
      scaleT = T.ScaleTransform(h=currHeight,w=currWidth,new_h=vgg_im_size,new_w=vgg_im_size,interp="nearest") 
      image = scaleT.apply_image(image.copy())

      # Apply the scaling to the bbox
      # THIS IS HANDLED IN annotations_to_instances, so long as this is appended to the list of transforms

      # Add this to the list of transforms
      transforms = transforms + scaleT

      # Set the dimensions
      dataset_dict["height"] = image.shape[0]
      dataset_dict["width"]  = image.shape[1]
    # not vgg or yolo
    else:# Downscale only at this threshold
      # Downscale only at this threshold
      if(currHeight > thresholdDimension or currWidth > thresholdDimension):
        myNewH = 0
        myNewW = 0
        # Scale the longest dimension to 1333, the shorter to 800
        if(currHeight > currWidth): 
          myNewH = thresholdDimension
          ratio = currHeight/float(myNewH)
          myNewW = currWidth/float(ratio)
          myNewW = int(round(myNewW))
          # myNewW = 800
        else:
          # myNewH = 800
          myNewW = thresholdDimension
          ratio = currWidth/float(myNewW)
          myNewH = currHeight/float(ratio)
          myNewH = int(round(myNewH))

        # Apply the scaling transform
        scaleT = T.ScaleTransform(h=currHeight,w=currWidth,new_h=myNewW,new_w=myNewH,interp="nearest") 
        image = scaleT.apply_image(image.copy())

        # Apply the scaling to the bbox
        # THIS IS HANDLED IN annotations_to_instances, so long as this is appended to the list of transforms

        # Add this to the list of transforms
        transforms = transforms + scaleT

        # Set the dimensions
        dataset_dict["height"] = image.shape[0]
        dataset_dict["width"]  = image.shape[1]
    
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

    dataset_dict["transforms"] = transforms

    return dataset_dict

  # # Small mappers
  # def small_train_mapper(self,dataset_dict):
  #   return self.train_mapper(dataset_dict,"small")

  # def small_test_mapper(self,dataset_dict):
  #   if(self.is_test_time_mapping):
  #     return self.test_mapper(dataset_dict,"small")
  #   else:
  #     return self.train_mapper(dataset_dict,"small")

  # # Large mappers
  # def large_train_mapper(self,dataset_dict):
  #   return self.train_mapper(dataset_dict,"large")

  # def large_test_mapper(self,dataset_dict):
  #   if(self.is_test_time_mapping):
  #     return self.test_mapper(dataset_dict,"large")
  #   else:
  #     return self.train_mapper(dataset_dict,"large")

  # # Full mappers
  # def full_train_mapper(self,dataset_dict):
  #   return self.train_mapper(dataset_dict,"full")

  # def full_test_mapper(self,dataset_dict):
  #   if(self.is_test_time_mapping):
  #     return self.test_mapper(dataset_dict,"full")
  #   else:
  #     return self.train_mapper(dataset_dict,"full")

  # # Comparison mappers
  # def comparison_train_mapper(self,dataset_dict):
  #   return self.train_mapper(dataset_dict,"comparison")

  # def comparison_test_mapper(self,dataset_dict):
  #   if(self.is_test_time_mapping):
  #     return self.test_mapper(dataset_dict,"comparison")
  #   else:
  #     return self.train_mapper(dataset_dict,"comparison")