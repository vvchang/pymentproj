#!/bin/bash
#SBATCH --job-name=pyment_motion
#SBATCH --output=pyment_motion_%j.out
#SBATCH --error=pyment_motion_%j.err
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=bios26211
#SBATCH --mail-type=ALL  # Email notification options: ALL, BEGIN, END, FAIL, ALL, NONE
#SBATCH --mail-user=enzezheng@rcc.uchicago.edu  

module load apptainer/1.4.1

mkdir -p pyment-public/data/headmotion/outputs

apptainer run --nv \
--bind pyment-public/data/headmotion/raw:/input \
--bind pyment-public/data/headmotion/outputs:/output \
--bind pyment-public/licenses:/licenses \
pyment_latest.sif 