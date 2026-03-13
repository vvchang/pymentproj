#!/bin/bash
#SBATCH --job-name=pyment_nomotion
#SBATCH --output=pyment_nomotion_%j.out
#SBATCH --error=pyment_nomotion_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=bios26211

module load apptainer/1.4.1

mkdir -p pyment-public/data/nomotion/outputs

apptainer run --nv \
--bind pyment-public/data/nomotion/raw:/input \
--bind pyment-public/data/nomotion/outputs:/output \
--bind pyment-public/licenses:/licenses \
pyment_latest.sif 