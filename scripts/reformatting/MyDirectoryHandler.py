def GetDatasetDirectories(dataset_used):
  if(dataset_used == "small"):
    baseOutputDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/outputs/small/"
    baseDirectory       = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/small_set/photos/"
    trainDirectory      = baseDirectory + "train/"
    valDirectory        = baseDirectory + "val/"
    imageDirectory      = baseDirectory + "images/"
    sourceJsonDirectory = baseDirectory + "data.json"
  elif(dataset_used == "large"):
    baseOutputDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/outputs/large/"
    baseDirectory       = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/large_set/"
    trainDirectory      = baseDirectory + "train/"
    valDirectory        = baseDirectory + "val/"
    imageDirectory      = baseDirectory + "images/"
    sourceJsonDirectory = baseDirectory + "data.json"
  elif(dataset_used == "full"):
    baseOutputDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/outputs/full/"
    baseDirectory       = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/full_set/"
    trainDirectory      = baseDirectory + "train/"
    valDirectory        = baseDirectory + "val/"
    imageDirectory      = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/large_set/" + "images/"
    sourceJsonDirectory = baseDirectory + "data.json"
  elif(dataset_used == "split0"):
    baseOutputDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/outputs/full/"
    baseDirectory       = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/full_set/splits/0_split/"
    trainDirectory      = baseDirectory + "train/"
    valDirectory        = baseDirectory + "val/"
    imageDirectory      = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/large_set/" + "images/"
    sourceJsonDirectory = baseDirectory + "data.json"
  elif(dataset_used == "split1"):
    baseOutputDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/outputs/full/"
    baseDirectory       = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/full_set/splits/1_split/"
    trainDirectory      = baseDirectory + "train/"
    valDirectory        = baseDirectory + "val/"
    imageDirectory      = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/large_set/" + "images/"
    sourceJsonDirectory = baseDirectory + "data.json"
  elif(dataset_used == "split2"):
    baseOutputDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/outputs/full/"
    baseDirectory       = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/full_set/splits/2_split/"
    trainDirectory      = baseDirectory + "train/"
    valDirectory        = baseDirectory + "val/"
    imageDirectory      = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/large_set/" + "images/"
    sourceJsonDirectory = baseDirectory + "data.json"
  elif(dataset_used == "split3"):
    baseOutputDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/outputs/full/"
    baseDirectory       = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/full_set/splits/3_split/"
    trainDirectory      = baseDirectory + "train/"
    valDirectory        = baseDirectory + "val/"
    imageDirectory      = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/large_set/" + "images/"
    sourceJsonDirectory = baseDirectory + "data.json"
  elif(dataset_used == "split4"):
    baseOutputDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/outputs/full/"
    baseDirectory       = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/full_set/splits/4_split/"
    trainDirectory      = baseDirectory + "train/"
    valDirectory        = baseDirectory + "val/"
    imageDirectory      = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/large_set/" + "images/"
    sourceJsonDirectory = baseDirectory + "data.json"
  elif(dataset_used == "comparison"):
    baseOutputDirectory = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/outputs/comparison/"
    baseDirectory       = "/mnt/storage/home/ja16475/sharks/detectron2/scratch/comparison_set/"
    trainDirectory      = baseDirectory + "train/"
    valDirectory        = baseDirectory + "val/"
    imageDirectory      = baseDirectory + "images/"
    sourceJsonDirectory = baseDirectory + "data.json"
  else:
    raise ValueError("dataset_used doesn't exist:"+dataset_used)

  return baseOutputDirectory, baseDirectory, trainDirectory, valDirectory, imageDirectory, sourceJsonDirectory
