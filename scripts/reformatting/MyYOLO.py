import torch
from torch import nn
from torchvision import transforms
import YoloModels
import yolo.utils.utils
from yolo.utils.utils import build_targets, to_cpu, non_max_suppression
from detectron2.structures import Instances
from detectron2.structures import Boxes
import os

class MyYOLO(YoloModels.Darknet):
  def __init__(self,config_path):
    super().__init__(config_path)
    self.device = torch.device("cuda")
    self.to(self.device)
    self.conf_threshold = 0.8
    self.nms_threshold  = 0.4
    # self.normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    #  std=[0.229, 0.224, 0.225])

  # Take in a batch of inputs
  def forward(self, batched_inputs):
    # Batched inputs is a list of mapped dictionaries
    # Note that this depends on the shape being 224x224: this is handled in the mapper
    # print(batched_inputs[0].keys())

    # Get out the images
    batched_images = [b["image"] for b in batched_inputs]

    # Normalise (required for yolo)
    # batched_images = [self.normalise(im) for im in batched_images]

    # Stack into one big tensor
    images_tensor = torch.stack(batched_images)

    # Vgg forward takes in a tensor, get out some logits
    # self.yolov3_model = self.yolov3_model.to(device)
    
    self.to(self.device)
    images_tensor = images_tensor.to(self.device)

    # print(type(images_tensor))
    # print(images_tensor.shape)
    # print(images_tensor)
  

    if self.training:
      # Get the height and widths
      batched_images_w_h = [ (b["width"],b["height"]) for b in batched_inputs]

      # batched_images_w_h = [ (b["image"].shape[1],b["image"].shape[1]) for b in batched_inputs]
      # Get the target classes
      target_classes = [b["classID"] for b in batched_inputs]
      # Compute the bboxes
      target_bboxes = [b["instances"] for b in batched_inputs]
      # print(target_bboxes)
      target_bboxes = [b.get("gt_boxes") for b in target_bboxes]
      target_centers = [b.get_centers().tolist()[0] for b in target_bboxes]
      target_bboxes = [b.tensor.tolist()[0] for b in target_bboxes]
      target_w_h = []
      for b in target_bboxes:
        x0,y0,x1,y1 = b
        w = abs(x0-x1)
        h = abs(y0-y1)
        target_w_h.append((w,h))
      
      target_bboxes = []
      for bbox_center,bbox_w_h,img_w_h,target_class in zip(target_centers,target_w_h,batched_images_w_h,target_classes):
        img_width  = img_w_h[0]
        img_height = img_w_h[1]
        center_x = bbox_center[0] / img_width
        center_y = bbox_center[1] / img_height
        width  = bbox_w_h[0] / img_width
        height = bbox_w_h[1] / img_height
        target_bboxes.append((0,target_class,center_x,center_y,width,height ))

      targets_tensor = torch.tensor(target_bboxes).to(self.device)

      # targets need to be in form:
      # tensor([[0.0000, 0.0000, 0.4900, 0.5000, 0.1454, 0.1829]])
      # index_of_bbox_in_image?, label_idx x_center y_center width height
      # The coordinates should be scaled [0, 1]

      # print(images_tensor.shape)
      # print(targets_tensor.shape)

      losses,_ = super().forward(images_tensor,targets_tensor)


      return {"total_loss" : losses}
    else:
      with torch.no_grad():
        # Forward pass the model
        outputs = super().forward(images_tensor)
        # nms
        outputs = non_max_suppression(outputs, conf_thres=self.conf_threshold, nms_thres=self.nms_threshold)

        # For each output,batchedim (Note: only 1 image per batch in evaluation)
        for output,batched_input in zip(outputs,batched_inputs):
          # height,width
          im_height = batched_input["height"]
          im_width  = batched_input["width"]

          # Get out predictions
          try:
            pred_boxes   = output[:,:4]
            pred_scores  = output[:, 4]
            pred_classes = output[:,-1].int()
          except:
            new_instance = Instances((im_height,im_width))
            new_instance.pred_boxes   = Boxes(torch.tensor([]))
            new_instance.scores       = torch.tensor([])
            new_instance.pred_classes = torch.tensor([]).int()
            return [{"instances" : new_instance}]

          # a "box" is len 4: center_x,center_y,width,height
          # scaled between 0 and 1

          pred_boxes = Boxes(pred_boxes)

          # Add the predictions
          new_instance = Instances((im_height,im_width))
          new_instance.pred_boxes   = pred_boxes
          new_instance.scores       = pred_scores
          new_instance.pred_classes = pred_classes
        
          # Immediately return for this loop as testing only involves 1 batch
          return [{"instances" : new_instance}]


def Create_YOLO(num_classes):
  # Create the model from config
  # Small set
  if(num_classes == 121):
    config_path = "/yolo/config/small-set-config.cfg"
  # Comparison set
  elif(num_classes == 85):
    # config_path = os.getcwd()+ "/yolo/config/" + "comparison-set-config.cfg"
    config_path = "/yolo/config/comparison-set-config.cfg"
  # Large or full
  elif(num_classes == 304):
    # config_path = "/sharks/detectron2/yolo/config/" + "large-full-set-config.cfg"
    config_path = "/yolo/config/large-full-set-config.cfg"
  else:
    raise ValueError("Create_YOLO: Num classes: " + str(num_classes) + " does not exist as a config")
  yolo_model = MyYOLO(config_path)
  # Init weights
  yolo_model.apply(yolo.utils.utils.weights_init_normal)
  # Load pre_trained_weights
  pretrained_weights = "/mnt/storage/home/ja16475/sharks/detectron2/yolo/darknet53.conv.74"
  yolo_model.load_darknet_weights(pretrained_weights)
  
  # Return model
  return yolo_model.to(torch.device("cuda"))
