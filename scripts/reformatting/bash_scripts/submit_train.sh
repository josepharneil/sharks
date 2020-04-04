#!/bin/bash

# Default values of arguments
LR=-1
MODEL=-1
MAX_ITER=-1
QUEUE=0
ACC=0
RESUME=0
BATCHSIZE=0
THRESHOLD=800
TESTTIME=1
DATASET=2
OTHER_ARGUMENTS=()

# Loop through arguments and process them
for arg in "$@"
do
    case $arg in
	-d|--dataset)
        DATASET="$2"
        shift # Remove argument name from processing
        shift # Remove argument value from processing
        ;;
        -lr|--learning-rate)
        LR="$2"
        shift # Remove argument name from processing
        shift # Remove argument value from processing
        ;;
	-m|--model)
        MODEL="$2"
        shift # Remove argument name from processing
        shift # Remove argument value from processing
        ;;
	-i|--max-iter)
        MAX_ITER="$2"
        shift # Remove argument name from processing
        shift # Remove argument value from processing
        ;;
	-s|--short)
        QUEUE=1
        shift # Remove argument name from processing
        # shift # Remove argument value from processing
        ;;
	-a|--accuracy)
        ACC=1
        shift # Remove argument name from processing
        # shift # Remove argument value from processing
        ;;
	-r|--resume)
        RESUME="$2"
	shift # Remove argument name from processing
        shift # Remove argument value from processing
        ;;
	-b|--batch-size)
        BATCHSIZE="$2"
        shift # Remove argument name from processing
        shift # Remove argument value from processing
        ;;
	-t|--threshold)
        THRESHOLD="$2"
        shift # Remove argument name from processing
        shift # Remove argument value from processing
        ;;
	-tt|--test-time)
        TESTTIME="$2"
        shift # Remove argument name from processing
        shift # Remove argument value from processing
        ;;
    esac
done

if [ $QUEUE -eq 0 ]
then
  echo Using the standard queue
  echo Training dataset $DATASET
  sbatch train_long.sh -lr $LR -m $MODEL -i $MAX_ITER -a $ACC -r $RESUME -b $BATCHSIZE -t $THRESHOLD -tt $TESTTIME -d $DATASET
else
  echo Using the very short queue
  echo Training dataset $DATASET
  sbatch train_short.sh -lr $LR -m $MODEL -i $MAX_ITER -a $ACC -r $RESUME -b $BATCHSIZE -t $THRESHOLD -tt $TESTTIME -d $DATASET
fi
