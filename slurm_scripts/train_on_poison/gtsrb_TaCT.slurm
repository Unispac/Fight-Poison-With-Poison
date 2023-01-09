#!/bin/bash
#SBATCH --job-name=normal    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --time=2:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=logs/gtsrb/poison_train/TaCT.out
#SBATCH --error=logs/gtsrb/poison_train/TaCT.err

module purge
module load anaconda3/2022.5
conda activate py38

python -u train_on_poisoned_set.py -dataset gtsrb -poison_type TaCT -poison_rate 0.005 -cover_rate 0.005 -seed 666