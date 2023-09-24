#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8gb

source ~/.bashrc

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module load compiler/gnu
module load mpi/openmpi
conda activate ips_apax

dvc repro