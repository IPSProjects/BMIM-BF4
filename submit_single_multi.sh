#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb

source ~/.bashrc

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export IPSUITE_CP2K_SHELL="mpirun -n ${SLURM_NTASKS} cp2k_shell.psmp"

module load compiler/gnu
module load mpi/openmpi
conda activate ips_apax

dvc repro
