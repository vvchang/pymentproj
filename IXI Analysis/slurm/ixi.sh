#!/bin/bash
#SBATCH --job-name=pyment_testall
#SBATCH --output=pyment_testall_%j.out
#SBATCH --error=pyment_testall_%j.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=bios26211
#SBATCH --mail-type=ALL  # Email notification options: ALL, BEGIN, END, FAIL, ALL, NONE
#SBATCH --mail-user=vchang5@rcc.uchicago.edu 

module load apptainer/1.4.1

mkdir -p pyment-public/data/ixi_all/outputs

apptainer run --nv \
--bind pyment-public/data/ixi_all/raw:/input \
--bind pyment-public/data/ixi_all/outputs:/output \
--bind pyment-public/licenses:/licenses \
pyment_latest.sif 