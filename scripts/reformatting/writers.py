import torch

from detectron2.utils.events import EventWriter
from detectron2.utils.events import get_event_storage

from detectron2.utils.events import JSONWriter

import evaluate
import datetime
from termcolor import colored, cprint

class TensorboardAndLogWriter(EventWriter):
  """
  Write all scalars to a tensorboard file.
  """

  def __init__(self, cfg, dataset_used, max_iter: int, log_dir: str, window_size: int = 20 ,**kwargs):
    """
    Args:
        log_dir (str): the directory to save the output events
        window_size (int): the scalars will be median-smoothed by this window size

        kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
    """
    self._window_size = window_size
    from torch.utils.tensorboard import SummaryWriter
    # self._file_handle = PathManager.open(json_file, "a")

    self._writer = SummaryWriter(log_dir, **kwargs)

    self._max_iter = max_iter

    # self.evaluator = evaluate.MyEvaluator(cfg,dataset_used)

    # if(self.parser.parse_args().accuracy == 1):
      # self.myEvaluator = evaluate.MyEvaluator(model,cfg,dataset_used)


  def write(self):
    # Get the storage
    storage = get_event_storage()

    # Evaluate accuracy
    # if(self.parser.parse_args().accuracy == 0):
    #   result_train = {}
    #   result_train["accuracy"] = -1
    #   result_test = {}
    #   result_test["accuracy"] = -1
    # else:
    #   result_train = self.evaluator.EvaluateTrainTopKAccuracy(model,1,isReturn=True)
    #   result_test  = self.evaluator.EvaluateTestTopKAccuracy(model,1,isReturn=True)

    # Add accuracy scalar
    # print(result_train["accuracy"])
    # print(result_train["accuracy"]*100)
    # print(round(result_train["accuracy"]*100,2))
    # accuracy_train = round((result_train["accuracy"]*100),2)
    # accuracy_test  = round((result_test["accuracy"]*100 ),2)
    
    # storage.put_scalar("accuracy_train",accuracy_train,smoothing_hint=False)
    # storage.put_scalar("accuracy_test" ,accuracy_test ,smoothing_hint=False)
    # accuracy_test = -1

    # self._writer.add_scalar("accuracy_train", accuracy_train, storage.iter)
    # self._writer.add_scalar("accuracy_test", accuracy_test, storage.iter)

    # to_save = {"iteration": storage.iter}
    # to_save.update(storage.latest_with_smoothing_hint(self._window_size))
    # self._file_handle.write(json.dumps(to_save, sort_keys=True) + "\n")
    # self._file_handle.flush()
    # try:
    #     os.fsync(self._file_handle.fileno())
    # except AttributeError:
    #     pass



    for k, v in storage.latest_with_smoothing_hint(self._window_size).items():
      # if(k != "accuracy")
      self._writer.add_scalar(k, v, storage.iter)
      # print(k,v,storage.iter)
      # if(k == "accuracy_shark_train"): accuracy_train = v
      # if(k == "accuracy_shark_val"):   accuracy_test  = v

    if len(storage.vis_data) >= 1:
      for img_name, img, step_num in storage.vis_data:
        self._writer.add_image(img_name, img, step_num)
      storage.clear_images()


    # Below here: logging
    # storage = get_event_storage()
    iteration = storage.iter

    data_time, time = None, None
    eta_string = "N/A"
    try:
        data_time = storage.history("data_time").avg(20)
        time = storage.history("time").global_avg()
        eta_seconds = storage.history("time").median(1000) * (self._max_iter - iteration)
        storage.put_scalar("eta_seconds", eta_seconds, smoothing_hint=False)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        # accuracy_string = "acc"
    except KeyError:  # they may not exist in the first few iterations (due to warmup)
        pass

    try:
        lr = "{:.6f}".format(storage.history("lr").latest())
    except KeyError:
        lr = "N/A"

    if torch.cuda.is_available():
        max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
    else:
        max_mem_mb = None

    lossesString ="  ".join(["{}: {:.3f}".format(k, v.median(20))
                      for k, v in storage.histories().items()
                      if "loss" in k])
                      
    timeString = "time: {:.4f}".format(time) if time is not None else ""
    data_time_string = "data_time: {:.4f}".format(data_time) if data_time is not None else ""
    memoryString = "max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else ""

    # Create the accuracy string
    try:
      accuracy_train = storage.history("accuracy_shark_train").latest()
      accuracy_test = storage.history("accuracy_shark_val").latest()
      train_accuracy_string = str(accuracy_train)
      test_accuracy_string  = str(accuracy_test)
    except:
      train_accuracy_string = "unevaluated"
      test_accuracy_string  = "unevaluated"
    
    # testing
    # accuracy_string = "temp"
    # storage.put_scalar("accuracy",0)

    logRedString = "eta: "+eta_string+"  iter: "+str(iteration)
    # logString = "eta: "+eta_string+"  iter: "+str(iteration) +"  " +str(lossesString)+"  accuracy: "+accuracy_string+"  "+timeString+"  "+data_time_string+"  "+"lr: "+str(lr)+"  "+memoryString
    logString = "  " +str(lossesString)+"  train accuracy: "+train_accuracy_string +"  test accuracy: "+test_accuracy_string +"  "+timeString+"  "+data_time_string+"  "+"lr: "+str(lr)+"  "+memoryString
    print(colored(logRedString,"red")+logString)


  def close(self):
    # self._file_handle.close()
    if hasattr(self, "_writer"):  # doesn't exist when the code fails at import
      self._writer.close()
