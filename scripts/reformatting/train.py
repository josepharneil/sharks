# -*- coding: utf-8 -*-
"""shark-initialseti.pynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oUGEC-RPu9DxnDLidVM2su46DVpjjnpV

# Installs and Imports
"""

# print("Importing...")

import torch, torchvision
import detectron2
import os
from collections import OrderedDict
import logging
from detectron2.utils.events import get_event_storage
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import DatasetEvaluator, print_csv_format

import mappers
import writers
import evaluate
import DefaultTrain

from termcolor import colored, cprint

from detectron2.utils import comm
from detectron2.engine import hooks
from detectron2.config import CfgNode
from typing import Any, Dict, List, Set

def my_build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module in model.modules():
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if isinstance(module, norm_module_types):
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
            elif key == "bias":
                # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                # hyperparameters are by default exactly the same as for regular
                # weights.
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
    # Adam(params, lr, betas, eps, weight_decay, amsgrad)
    # default momentum: 0.9
    # optimizer = torch.optim.Adam(params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
    # optimizer = torch.optim.Adam(params, cfg.SOLVER.BASE_LR)
    # print("Using ADAM optimizer - note that the momentum is the default in ADAM, and is not associated with the CFG")
    return optimizer



# class SmallSetTrainer(DefaultTrain.MyDefaultTrainer):
class My_Trainer(DefaultTrain.MyDefaultTrainer):
  def __init__(self,cfg,parser,getter,dataset_used,threshold_dimension,is_test_time_mapping,modelLink):
    self.getter = getter
    self.dataset_used = dataset_used
    self.mapper_object = mappers.My_Mapper(dataset_used,threshold_dimension,is_test_time_mapping,modelLink)
    super().__init__(cfg,parser,self.mapper_object)

  # @classmethod
  # def build_test_loader(cls, cfg, dataset_name):
    # return build_detection_test_loader(cfg, dataset_name, mapper=mapper_object.test_mapper)
    # return build_detection_test_loader(cfg, dataset_name, mapper=mapper_object.test_mapper)
    # return build_detection_train_loader(cfg, mapper=self.mapper_object.test_mapper)

  # @classmethod
  # def build_train_loader(cls, cfg):
    # return build_detection_train_loader(cfg, mapper=mapper_object.train_mapper)
    # return build_detection_train_loader(cfg, mapper=self.mapper_object.train_mapper)

  # @classmethod
  def build_writers(self):
    """
    Build a list of writers to be used. By default it contains
    writers that write metrics to the screen,
    a json file, and a tensorboard event file respectively.
    If you'd like a different list of writers, you can overwrite it in
    your trainer.

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.

    It is now implemented by:

    .. code-block:: python

        return [
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    """
    # Assume the default print/log frequency.
    return [
        writers.JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
        # writers.TensorboardAndLogWriter(self.cfg,"small",self.max_iter,self.cfg.OUTPUT_DIR+"/tensorboard"),
        writers.TensorboardAndLogWriter(self.cfg,self.dataset_used,self.max_iter,self.cfg.OUTPUT_DIR+"/tensorboard"),
    ]

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, getter, dataset_used):
      """
      Returns:
          DatasetEvaluator or None

      It is not implemented by default.
      """
      if(dataset_name == "shark_val"):
        return evaluate.TopKAccuracy(getter=getter, dataset_used=dataset_used, cfg=cfg, k=1, output_images=True)
      else:
        return evaluate.TopKAccuracy(getter=getter, dataset_used=dataset_used, k=1)
      

      raise NotImplementedError(
          "Please either implement `build_evaluator()` in subclasses, or pass "
          "your evaluator as arguments to `DefaultTrainer.test()`."
      )

  @classmethod
  def test(cls, cfg, model, getter, dataset_used, mapper_object, evaluators=None):
      """
      Args:
          cfg (CfgNode):
          model (nn.Module):
          evaluators (list[DatasetEvaluator] or None): if None, will call
              :meth:`build_evaluator`. Otherwise, must have the same length as
              `cfg.DATASETS.TEST`.

      Returns:
          dict: a dict of result metrics
      """
      logger = logging.getLogger(__name__)
      if isinstance(evaluators, DatasetEvaluator):
          evaluators = [evaluators]
      if evaluators is not None:
          assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
              len(cfg.DATASETS.TEST), len(evaluators)
          )

      results = OrderedDict()
      for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
          # if(not isTrackAccuracy):
            # break
          data_loader = cls.build_test_loader(cfg, dataset_name,mapper_object)
          # When evaluators are passed in as arguments,
          # implicitly assume that evaluators can be created before data_loader.
          if evaluators is not None:
              evaluator = evaluators[idx]
          else:# If the passed in arg evaluators is NONE
              try:
                  evaluator = cls.build_evaluator(cfg, dataset_name, getter, dataset_used)
              except NotImplementedError:
                  logger.warn(
                      "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                      "or implement its `build_evaluator` method."
                  )
                  results[dataset_name] = {}
                  continue
          printString = "Evaluating top 1 accuracy of: " + dataset_name
          print(colored(printString,"yellow"))
          results_i = evaluate.inference_on_dataset(model, data_loader, evaluator)
          printString = "Done evaluating top 1 accuracy of: " + dataset_name
          print(colored(printString,"yellow"))

          topkacc = results_i["topKAcc"]
          accuracy_test  = round(  (topkacc["accuracy"]*100),2  )
          
          storage = get_event_storage()
          storage.put_scalar("accuracy_"+dataset_name,accuracy_test ,smoothing_hint=False)
          # iteration = storage.iter
          # logRedString = "eta: "+eta_string+"  iter: "+str(iteration)
          # logString = "  " +str(lossesString) +"  "+timeString+"  "+data_time_string+"  "+"lr: "+str(lr)+"  "+memoryString
          # print(colored(logRedString,"red")+logString)

          results_i["topKAcc"] = {  "total_num":   float(topkacc["total_num"])   , 
                                    "num_correct": float(topkacc["num_correct"]) , 
                                    "accuracy":    float(topkacc["accuracy"])    , 
                                    "k":           float(topkacc["k"])           }#,
                                    #"perClass":    float(topkacc["perClass"])      }

          AccToWrite = str(round((float(topkacc["accuracy"])*100),2))
          StrToWrite = dataset_name + ": " + AccToWrite
          text_file = open(cfg.OUTPUT_DIR+"/parameters-information.txt","a+")
          text_file.write(StrToWrite)
          text_file.close()

          results[dataset_name] = results_i
          if comm.is_main_process():
              assert isinstance(
                  results_i, dict
              ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                  results_i
              )
              logger.info("Evaluation results for {} in csv format:".format(dataset_name))
              print_csv_format(results_i)

      if len(results) == 1:
          results = list(results.values())[0]
      return results


  def build_hooks(self):
    """
    Build a list of default hooks, including timing, evaluation,
    checkpointing, lr scheduling, precise BN, writing events.

    Returns:
        list[HookBase]:
    """
    cfg = self.cfg.clone()
    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

    ret = \
    [
      hooks.IterationTimer(),
      hooks.LRScheduler(self.optimizer, self.scheduler),
      hooks.PreciseBN(
          # Run at the same freq as (but before) evaluation.
          cfg.TEST.EVAL_PERIOD,
          self.model,
          # Build a new data loader to not affect training
          self.build_train_loader(cfg,self.mapper_object),
          cfg.TEST.PRECISE_BN.NUM_ITER,
      )
      if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
      else None,
    ]

    # Do PreciseBN before checkpointer, because it updates the model and need to
    # be saved by checkpointer.
    # This is not always the best: if checkpointing has a different frequency,
    # some checkpoints may have more precise statistics than others.
    if comm.is_main_process():
      ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

    def test_and_save_results():
      # self._last_eval_results = self.test(self.cfg, self.model, self.isTrackAccuracy, self.getter, self.dataset_used)
      self._last_eval_results = self.test(self.cfg,self.model,self.getter,self.dataset_used,self.mapper_object,evaluators=None)
      return self._last_eval_results

    # Do evaluation after checkpointer, because then if it fails,
    # we can use the saved checkpoint to debug.
    ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

    if comm.is_main_process():
      numberOfSamples = 25
      step = -1
      if(self.max_iter <= numberOfSamples):
        # Eg, maxiter = 20, so step = 20/2 = 10, take a sample every 10
        step = int(round(float(self.max_iter)/float(2),2))
      else:
        # Eg 10000/20 = 500, so will take a sample every 500 iterations
        step = float(self.max_iter)/float(numberOfSamples)
        step = int(round(step,0))
        if(step < 1): step = 1

      # print("!!!!!!!!!!!!!!STEPS: ", step)
      # ret.append(hooks.PeriodicWriter(self.build_writers()))
      # run writers in the end, so that evaluation metrics are written
      ret.append(hooks.PeriodicWriter(self.build_writers(),period=step))
      # ret.append(hooks.PeriodicWriter(self.build_writers(),period=(self.max_iter-1)))
    return ret

  @classmethod
  def build_optimizer(cls, cfg, model):
    """
    Returns:
        torch.optim.Optimizer:

    It now calls :func:`detectron2.solver.build_optimizer`.
    Overwrite it if you'd like a different optimizer.
    """
    return my_build_optimizer(cfg, model)

  # @classmethod
  # def build_model(cls, cfg):
  #     """
  #     Returns:
  #         torch.nn.Module:

  #     It now calls :func:`detectron2.modeling.build_model`.
  #     Overwrite it if you'd like a different model.
  #     """
  #     model = build_model(cfg)
  #     logger = logging.getLogger(__name__)
  #     logger.info("Model:\n{}".format(model))
  #     return model




'''

class LargeSetTrainer(SmallSetTrainer):
  @classmethod
  def build_test_loader(cls, cfg, dataset_name, mapper_object):
    return build_detection_test_loader(cfg, dataset_name, mapper=mappers.large_test_mapper)

  @classmethod
  def build_train_loader(cls, cfg, mapper_object):
    return build_detection_train_loader(cfg, mapper=mappers.large_train_mapper)

  def build_writers(self):
    """
    Build a list of writers to be used. By default it contains
    writers that write metrics to the screen,
    a json file, and a tensorboard event file respectively.
    If you'd like a different list of writers, you can overwrite it in
    your trainer.

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.

    It is now implemented by:

    .. code-block:: python

        return [
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    """
    # Assume the default print/log frequency.
    return [
        writers.JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
        writers.TensorboardAndLogWriter(self.cfg,"large",self.max_iter,self.cfg.OUTPUT_DIR+"/tensorboard"),
    ]

class FullSetTrainer(SmallSetTrainer):
  @classmethod
  def build_test_loader(cls, cfg, dataset_name):
    return build_detection_test_loader(cfg, dataset_name, mapper=mappers.full_test_mapper)

  @classmethod
  def build_train_loader(cls, cfg):
    return build_detection_train_loader(cfg, mapper=mappers.full_train_mapper)

  def build_writers(self):
    """
    Build a list of writers to be used. By default it contains
    writers that write metrics to the screen,
    a json file, and a tensorboard event file respectively.
    If you'd like a different list of writers, you can overwrite it in
    your trainer.

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.

    It is now implemented by:

    .. code-block:: python

        return [
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    """
    # Assume the default print/log frequency.
    return [
        writers.JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
        writers.TensorboardAndLogWriter(self.cfg,"full",self.max_iter,self.cfg.OUTPUT_DIR+"/tensorboard"),
    ]



class ComparisonSetTrainer(SmallSetTrainer):
  @classmethod
  def build_test_loader(cls, cfg, dataset_name):
    return build_detection_test_loader(cfg, dataset_name, mapper=mappers.comparison_test_mapper)

  @classmethod
  def build_train_loader(cls, cfg):
    return build_detection_train_loader(cfg, mapper=mappers.comparison_train_mapper)

  def build_writers(self):
    """
    Build a list of writers to be used. By default it contains
    writers that write metrics to the screen,
    a json file, and a tensorboard event file respectively.
    If you'd like a different list of writers, you can overwrite it in
    your trainer.

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.

    It is now implemented by:

    .. code-block:: python

        return [
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    """
    # Assume the default print/log frequency.
    return [
        writers.JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
        writers.TensorboardAndLogWriter(self.cfg,"comparison",self.max_iter,self.cfg.OUTPUT_DIR+"/tensorboard"),
    ]


'''