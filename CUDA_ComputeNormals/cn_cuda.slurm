#!/bin/bash

#SBATCH --job-name ale_cuda
#SBATCH --error errors_%j.txt
#SBATCH --output output_%j.txt
#SBATCH --mail-user rossiale@dei.unipd.it
#SBATCH --mail-type END,FAIL
#SBATCH --partition allgroups
#SBATCH --ntasks 1
#SBATCH --mem 200M
#SBATCH --time 4:00
#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR

srun singularity exec --nv --bind .:/mnt/source,../CAPRI_Input:/mnt/Input,../CAPRI_Output:/mnt/Output  nvidia_pycuda_orig.sif /mnt/source/compute_normals.o