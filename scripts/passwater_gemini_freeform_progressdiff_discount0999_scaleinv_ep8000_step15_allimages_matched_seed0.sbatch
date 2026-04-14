#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=112G
#SBATCH --time=48:00:00
#SBATCH --job-name=passwat_pd999_h15_s0
#SBATCH --output=logs/%x-%j.out

export RLVLMF_TASK_ENV=softgym_PassWater
export RLVLMF_METHOD=progressdiff0999_scaleinv
export RLVLMF_EXP_NAME=progressdiff_discount0999_scaleinv_gemini_freeform_ep8000_step15_allimages_matched_seed0_passwater
export RLVLMF_USE_EPISODE=false
export RLVLMF_MAX_EPISODE_STEPS=15
export RLVLMF_IMAGE_REPLAY_CAPACITY=120000
export RLVLMF_NUM_TRAIN_STEPS=120000
export RLVLMF_NUM_SEED_STEPS=200
export RLVLMF_NUM_UNSUP_STEPS=1800
export RLVLMF_NUM_INTERACT=1000
export RLVLMF_EVAL_FREQUENCY=2000
export RLVLMF_SAVE_INTERVAL=12000
export RLVLMF_VIDEO_STEP_INTERVAL=12000
export RLVLMF_VIDEO_STEP_OFFSET=11985
export RLVLMF_REWARD_BATCH=100
export RLVLMF_REWARD_UPDATE=30
export RLVLMF_REWARD_LR=1e-4
export RLVLMF_MAX_FEEDBACK=20000
export RLVLMF_TERMINATE_ON_SUCCESS=false
export RLVLMF_RESNET=1

cd /project2/biyik_1165/RL-VLM-F_value-model
exec bash scripts/_run_matched_horizon_training.sh
