#!/bin/bash
#SBATCH --partition=single
#SBATCH --time=08:00:00
#SBATCH --mem=8gb
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate ips_apax

dvc repro
