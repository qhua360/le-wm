#!/bin/bash
#SBATCH --job-name=lewm-ogbench-cube-stereo-b06
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

# Stereo LeWM (6 cm baseline) on ogbench-cube, 4x H200 DDP, 100 epochs. Submit from the repo root:
#   sbatch slurm/train_ogbench_stereo.sh [hydra overrides...]
# Data: python render_stereo.py --src .../cube_single_expert.h5 --dst .../cube_single_expert_stereo_b06.h5

export RUN_NAME=lewm-ogbench-cube-stereo-b06
export DATA_CFG=ogb_stereo
export DATA_SRC=/scratch/ad158/stable-wm/datasets/ogbench/cube_single_expert_stereo_b06.h5
exec /home/ad158/le-wm/slurm/_train_lewm.sh model=lewm_stereo "$@"
