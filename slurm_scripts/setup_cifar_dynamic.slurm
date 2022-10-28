#!/bin/bash
#SBATCH --job-name=normal    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --time=0:10:00          # total run time limit (HH:MM:SS)

nohup python create_poisoned_set.py -dataset cifar10 -poison_type dynamic -poison_rate 0.01 > temp_log.out 2>&1 &