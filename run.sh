#!/bin/bash
#
#---------------------------------------------
# SLURM job script for single CPU/GPU
#---------------------------------------------
#
#
#SBATCH --job-name=ISTAnt_gen       # Job name
#SBATCH --output=logs/ISTAnt_%j.log # Output file
#SBATCH --time=08:00:00             # Maximum walltime
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --mem=10G                   # Memory allocation
#SBATCH --partition=gpu100          # Partition to use (e.g., gpu)
#SBATCH --gres=gpu:1                # Number of GPUs requested (adjust as needed)
#
# Load your environment or module
##source ~/.bashrc                  # Ensure you load your bash profile
module load conda
conda activate crl                  # Activate your conda environment

# general variables 
task="or"

# Run experiments
# srun python ./src/run_gen.py --sc all --task $task
# srun python ./src/run_gen.py --sc random_easy --task $task
srun python ./src/run_gen.py --sc experiment1 --task $task
# srun python ./src/run_gen.py --sc treatment0 --task $task
# srun python ./src/run_gen.py --sc treatment1 --task $task
# srun python ./src/run_gen.py --sc treatment2 --task $task
# srun python ./src/run_gen.py --sc experiment0 --task $task 
# srun python ./src/run_gen.py --sc random --task $task
# srun python ./src/run_gen.py --sc position --task $task 
# srun python ./src/run_gen.py --sc position_easy --task $task
# srun python ./src/run_gen.py --sc experiment_easy --task $task 