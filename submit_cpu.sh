#!/bin/bash
#SBATCH --partition=cpu-multi
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=16gb

source ~/.bashrc

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module load compiler/gnu
module load mpi/openmpi
conda activate ips_apax

dvc repro
