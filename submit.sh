#!/bin/bash
#SBATCH --partition=cpu-multi
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=32gb

source ~/.bashrc

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module load compiler/gnu
module load mpi/openmpi
conda activate ipsuite

dvc repro
