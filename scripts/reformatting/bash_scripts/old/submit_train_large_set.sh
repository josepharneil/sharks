#!/bin/bash

# Default values of arguments
LR=-1
MODEL=-1
MAX_ITER=-1
QUEUE=0
ACC=0
RESUME=0
OTHER_ARGUMENTS=()

# Loop through arguments and process them
for arg in "$@"
do
    case $arg in
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
    esac
done

if [ $QUEUE -eq 0 ]
then
  echo Using the standard queue
  sbatch train_large_set_long.sh -lr $LR -m $MODEL -i $MAX_ITER -r $RESUME -a $ACC
else
  echo Using the very short queue
  sbatch train_large_set_short.sh -lr $LR -m $MODEL -i $MAX_ITER -r $RESUME -a $ACC
fi

