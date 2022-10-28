#!/bin/bash
#SBATCH --job-name=cleanse    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=1G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --time=4:00:00          # total run time limit (HH:MM:SS)

nohup python poison_cleanser_iter.py -dataset cifar10 -poison_type adaptive_blend -poison_rate 0.001 -cover_rate 0.001 -devices 0 -debug_info > logs/cifar/adp_blend_0.001.out 2>&1 &