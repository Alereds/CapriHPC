#!/bin/bash

#SBATCH --job-name ale_cuda
#SBATCH --error errors_%j.txt
#SBATCH --output output_%j.txt
#SBATCH --mail-user rossiale@dei.unipd.it
#SBATCH --mail-type END,FAIL
#SBATCH --partition allgroups
#SBATCH --ntasks 1
#SBATCH --mem 1G
#SBATCH --time 00:02:00
# requesting 1 GPU; 
#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR

srun singularity run --bind Output:/mnt container.sif