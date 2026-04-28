#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --job-name=passwat_pd1shp1o2sm_h30_s0
#SBATCH --output=logs/%x-%j.out

export RLVLMF_TASK_ENV=softgym_PassWater
export RLVLMF_METHOD=progressdiff_discount1_scale_hplus1_over2
export RLVLMF_EXP_NAME=progressdiff_smooth_discount1_scale_hplus1_over2_gemini_freeform_ep8000_step30_allimages_matched_seed0_passwater
export RLVLMF_USE_SMOOTH_RELABEL=1
export RLVLMF_SMOOTH_WINDOW=9
export RLVLMF_USE_EPISODE=false
export RLVLMF_MAX_EPISODE_STEPS=30
export RLVLMF_IMAGE_REPLAY_CAPACITY=240000
export RLVLMF_NUM_TRAIN_STEPS=240000
export RLVLMF_NUM_SEED_STEPS=400
export RLVLMF_NUM_UNSUP_STEPS=3600
export RLVLMF_NUM_INTERACT=2000
export RLVLMF_EVAL_FREQUENCY=4000
export RLVLMF_SAVE_INTERVAL=2000
export RLVLMF_VIDEO_STEP_INTERVAL=4000
export RLVLMF_REWARD_BATCH=100
export RLVLMF_REWARD_UPDATE=30
export RLVLMF_REWARD_LR=1e-4
export RLVLMF_MAX_FEEDBACK=20000
export RLVLMF_TERMINATE_ON_SUCCESS=false
export RLVLMF_RESNET=1

cd /project2/biyik_1165/RL-VLM-F_value-model
exec bash scripts/_run_matched_horizon_training.sh
