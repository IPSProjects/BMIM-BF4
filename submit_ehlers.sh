#!/bin/bash
#SBATCH --partition=dgx
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --job-name=293K-8Pairs

source ~/.bashrc
conda activate ips

# dvc repro
zntrack run ipsuite.nodes.ApaxJaxMD --name depl_ApaxJaxMD
