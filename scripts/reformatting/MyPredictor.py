import MyBuildModel
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T


class MyPredictor(DefaultPredictor):
  def __init__(self, cfg):
    self.cfg = cfg.clone()  # cfg can be modified by model
    self.model = MyBuildModel.my_build_model(self.cfg) #build_model(self.cfg)
    self.model.eval()
    self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    checkpointer = DetectionCheckpointer(self.model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    self.transform_gen = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )

    self.input_format = cfg.INPUT.FORMAT
    assert self.input_format in ["RGB", "BGR"], self.input_format