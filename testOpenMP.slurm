#!/bin/bash

#SBATCH --job-name ale_openmp
#SBATCH --error errors_%j.txt
#SBATCH --output output_%j.txt
#SBATCH --mail-user rossiale@dei.unipd.it
#SBATCH --mail-type END,FAIL
#SBATCH --partition allgroups
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 8G
#SBATCH --time 00:01:00

export OMP_NUM_THREADS=8

srun singularity exec testImage.sif ./openmp_master.o
