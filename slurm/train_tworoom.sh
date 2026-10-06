#!/bin/bash
#SBATCH --job-name=lewm-tworoom
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --exclude=bg5u16g1
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=/scratch/ad158/baselines/le-wm/logs/%x-%j.out
#SBATCH --error=/scratch/ad158/baselines/le-wm/logs/%x-%j.err

# LeWM baseline on tworoom, 4x H100 DDP, 100 epochs. Submit from the repo root:
#   sbatch slurm/train_tworoom.sh [hydra overrides...]

export RUN_NAME=lewm-tworoom
export DATA_CFG=tworoom
export DATA_SRC=/scratch/ad158/stable-wm/datasets/tworoom.h5
exec /home/ad158/le-wm/slurm/_train_lewm.sh "$@"
