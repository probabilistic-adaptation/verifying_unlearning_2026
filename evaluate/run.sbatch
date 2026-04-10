#!/bin/bash
#SBATCH -J unlearn_exp6_
#SBATCH -o  unlearn_exp6_%j.out
#SBATCH -e  unlearn_exp6_%j.err
#SBATCH -c 1
#SBATCH -p gpu
#SBATCH -t 0-24:00
#SBATCH --mem=32G
#SBATCH -n 1
#SBATCH --gres=gpu:2

source ~/.bashrc # for conda initialization
module load python
conda activate my_env

cd $SLURM_SUBMIT_DIR

srun -c 1 python run_experiment.py