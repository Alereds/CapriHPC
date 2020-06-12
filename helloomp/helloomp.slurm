#!/bin/bash
#SBATCH -J helloopenmp
#SBATCH -o output_%j.txt
#SBATCH -e errors_%j.txt
#SBATCH -t 01:30:00
#SBATCH -n 1
# notice we're using the '-c' option
# to request for OMP threads
#SBATCH -c 8
#SBATCH -p allgroups
#SBATCH --mem 1G

# set this to what you asked with '-c'

export OMP_NUM_THREADS=8

srun ./helloomp.o
