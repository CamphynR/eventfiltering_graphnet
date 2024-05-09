#!/bin/bash
#SBATCH --job-name=train-clasifier
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32000
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00
#SBATCH --output=output_train.txt

srun python3 train.py
