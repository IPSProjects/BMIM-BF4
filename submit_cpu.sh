#!/bin/bash
#SBATCH --partition=cpu-multi
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=64gb

source ~/.bashrc

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export IPSUITE_CP2K_SHELL="mpirun -n ${SLURM_NTASKS} cp2k_shell.psmp"

module load compiler/gnu
module load mpi/openmpi
conda activate ips_jax

dvc repro
