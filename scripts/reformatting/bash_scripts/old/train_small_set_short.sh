#!/usr/bin/env bash
#SBATCH --partition gpu_veryshort
#SBATCH --time 0-01:00
#SBATCH --mail-type END
#SBATCH --account comsm0018
#SBATCH --mem 64GB
#SBATCH --gres gpu:1

echo The Job ID is $SLURM_JOBID

# get rid of any modules already loaded
# module purge
# load in the module dependencies for this script
# module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"

module load CUDA

source activate detectVenv3

# Default values of arguments
LR=-1
MODEL=-1
MAX_ITER=-1
ACC=0
RESUME=""
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
	-r|--resume)
        RESUME="$2"
        shift # Remove argument name from processing
        shift # Remove argument value from processing
        ;;
	-a|--accuracy)
        ACC=1
        shift # Remove argument name from processing
        ;;
    esac
done

python3 train_test.py -d s -lr $LR -m $MODEL -i $MAX_ITER -id $SLURM_JOBID -a $ACC
