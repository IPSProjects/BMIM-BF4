#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH --job-name=283K-32Pairs

source ~/.bashrc
conda activate ips

# dvc repro
zntrack run ipsuite.nodes.ApaxJaxMD --name depl_ApaxJaxMD
