#!/bin/bash
#SBATCH --job-name=lewm-reacher
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --exclude=bg5u16g1
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=/scratch/ad158/baselines/le-wm/logs/%x-%j.out
#SBATCH --error=/scratch/ad158/baselines/le-wm/logs/%x-%j.err

# LeWM baseline on reacher, 4x H200 DDP, 100 epochs. Submit from the repo root:
#   sbatch slurm/train_reacher.sh [hydra overrides...]

export RUN_NAME=lewm-reacher
export DATA_CFG=dmc
export DATA_SRC=/scratch/ad158/stable-wm/datasets/reacher.h5
exec /home/ad158/le-wm/slurm/_train_lewm.sh "$@"
