#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p C032M0512G
#SBATCH --qos=high
#SBATCH -J DataPreProcessing
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

python -u get_pre_data_MASS.py