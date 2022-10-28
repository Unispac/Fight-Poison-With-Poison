#!/bin/bash
#SBATCH --job-name=normal    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=32        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=2G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:2            # number of gpus per node
#SBATCH --time=4:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=logs/cifar/ct_cleanse/clean_label.out
#SBATCH --error=logs/cifar/ct_cleanse/clean_label.err

module purge
module load anaconda3/2022.5
conda activate py38

python -u ct_cleanser.py -dataset cifar10 -poison_type clean_label -poison_rate 0.005 -devices 0,1 -debug_info