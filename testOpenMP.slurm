#!/bin/bash

#SBATCH --job-name ale_openmp
#SBATCH --error errors_%j.txt
#SBATCH --output output_%j.txt
#SBATCH --mail-user rossiale@dei.unipd.it
#SBATCH --mail-type END,FAIL
#SBATCH --partition allgroups
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 200M
#SBATCH --time 00:02:00

cd $SLURM_SUBMIT_DIR

export OMP_NUM_THREADS=8

srun singularity run --bind Output:/mnt container.sif
