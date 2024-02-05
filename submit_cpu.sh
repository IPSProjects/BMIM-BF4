#!/bin/bash
#SBATCH --partition=cpu-multi
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --mem=32gb
#SBATCH --job-name=ASEMD_19

source ~/.bashrc

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export IPSUITE_CP2K_SHELL="mpirun -n ${SLURM_NTASKS} cp2k_shell.psmp"

module load compiler/gnu
module load mpi/openmpi
conda activate ips_jax

zntrack run ipsuite.nodes.ASEMD --name uncorr_AIMD_ASEMD_19

# uncorr_AIMD_ASEMD_12 failed