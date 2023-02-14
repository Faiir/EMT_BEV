#!/usr/bin/env bash
#SBATCH --job-name=beverse_motion_benchmark_test
#SBATCH --output=beverse_motion_benchmark%j.log
#SBATCH --error=beverse_motion_benchmark%j.err
#SBATCH --mail-user=kraussn@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
set -e

source /home/kraussn/switch-cuda.sh 11.3

source /home/kraussn/anaconda3/bin/activate /home/kraussn/anaconda3/envs/motion_detr

cd /home/kraussn/EMT_BEV/  # navigate to the directory if necessary


srun /home/kraussn/anaconda3/envs/motion_detr/bin/python3 /home/kraussn/EMT_BEV/full_benchmark.py