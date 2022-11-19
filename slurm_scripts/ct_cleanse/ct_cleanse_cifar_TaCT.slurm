#!/bin/bash
#SBATCH --job-name=normal    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:2            # number of gpus per node
#SBATCH --time=8:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=logs/cifar/ct_cleanse/TaCT.out
#SBATCH --error=logs/cifar/ct_cleanse/TaCT.err

module purge
module load anaconda3/2022.5
conda activate py38

python -u ct_cleanser.py -dataset cifar10 -poison_type TaCT -poison_rate 0.02 -cover_rate 0.01 -devices 0,1 -debug_info