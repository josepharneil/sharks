import torch
import torch.nn as nn
from torchvision.models import vgg
from torchvision.models import utils as models_utils
from torchvision import transforms
# from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.structures import Instances


class MyVGG(vgg.VGG):
  def __init__(self, features, num_classes=1000, init_weights=True):
    super().__init__(features, num_classes, init_weights)
    self.device = torch.device("cuda")
    self.to(self.device)
    self.normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    self.criterion = nn.CrossEntropyLoss()
    self.softmax = nn.Softmax(dim=0)

  def forward(self, batched_inputs):
    # Batched inputs is a list of mapped dictionaries
    # Note that this depends on the shape being 224x224: this is handled in the mapper
    # print(batched_inputs[0].keys())

    # Get out the images
    batched_images = [b["image"] for b in batched_inputs]
    target_list = [b["classID"] for b in batched_inputs]

    # Normalise (required for this vgg)
    # normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    # std=[0.229, 0.224, 0.225])
    batched_images = [self.normalise(im) for im in batched_images]

    # Stack into one big tensor
    batched_images = torch.stack(batched_images)

    # Put it on the device
    batched_images = batched_images.to(self.device)

    if self.training:
      # Forward pass of batched images
      # logits_tensor = self.vggModel(batched_images)
      logits_tensor = super().forward(batched_images)
      
      # Make sure logits are on device
      logits_tensor = logits_tensor.to(self.device)
      # Get the targets
      target_tensor = torch.tensor(target_list).to(self.device)
      target_tensor.requires_grad = False

      # Compute the losses
      losses = self.criterion(logits_tensor,target_tensor)

      # Return the losses as a total
      return {"total_loss" : losses}
    else:
      with torch.no_grad():
        # Fwd through
        logits_tensor = super().forward(batched_images)
        logits_tensor = logits_tensor.to(self.device)

        # instances = []
        # For each image in batch (batch size is 1 for evaluation though)
        for i,logits in enumerate(logits_tensor):
          # Get the h,w
          w,h = batched_inputs[i]["width"],batched_inputs[i]["height"]
          # Convert the logits to probs
          probs = self.softmax(logits)
          # Get the highest prediction
          pred_confidence,pred_class = probs.max(0)

          # Add the highest prediction
          new_instance = Instances((h,w))
          new_instance.scores = torch.tensor([pred_confidence])
          new_instance.pred_classes = torch.tensor([pred_class])

          # instances.append(new_instance)
        
          return [{"instances" : new_instance}]


def my_vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = MyVGG(vgg.make_layers(vgg.cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = models_utils.load_state_dict_from_url(vgg.model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def my_vgg19_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return my_vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)


def Create_VGG(num_classes):
  v = my_vgg19_bn(pretrained=True)
  v.classifier[6] = nn.Linear(in_features=4096,out_features=num_classes,bias=True)
  for i,param in enumerate(v.parameters()):
    if(i < 30):
      # print(i,param)
      param.requires_grad = False
  return v.to(torch.device("cuda"))