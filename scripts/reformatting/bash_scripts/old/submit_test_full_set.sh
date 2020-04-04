#!/bin/bash

# Default values of arguments
MODEL=-1
QUEUE=0
OTHER_ARGUMENTS=()

# Loop through arguments and process them
for arg in "$@"
do
    case $arg in
	-m|--model)
        MODEL="$2"
        shift # Remove argument name from processing
        shift # Remove argument value from processing
        ;;
	-s|--short)
        QUEUE=1
        shift # Remove argument name from processing
        # shift # Remove argument value from processing
        ;;
    esac
done

if [ $QUEUE -eq 0 ]
then
  echo Using the standard queue
  sbatch test_full_set_long.sh -m $MODEL
else
  echo Using the very short queue
  sbatch test_full_set_short.sh -m $MODEL
fi

