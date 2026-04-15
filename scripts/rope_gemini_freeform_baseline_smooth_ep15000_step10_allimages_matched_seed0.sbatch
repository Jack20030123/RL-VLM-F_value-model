#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=rope_bsmooth_h10_s0
#SBATCH --output=logs/%x-%j.out

export RLVLMF_TASK_ENV=softgym_RopeFlattenEasy
export RLVLMF_METHOD=baseline
export RLVLMF_EXP_NAME=baseline_smooth_gemini_freeform_ep15000_step10_allimages_matched_seed0_rope
export RLVLMF_USE_SMOOTH_RELABEL=1
export RLVLMF_USE_EPISODE=true
export RLVLMF_MAX_EPISODE_STEPS=10
export RLVLMF_IMAGE_REPLAY_CAPACITY_EPISODES=15000
export RLVLMF_NUM_TRAIN_EPISODES=15000
export RLVLMF_NUM_SEED_EPISODES=25
export RLVLMF_NUM_UNSUP_EPISODES=225
export RLVLMF_NUM_INTERACT_EPISODES=125
export RLVLMF_EVAL_EPISODE_FREQUENCY=250
export RLVLMF_SAVE_EPISODE_INTERVAL=200
export RLVLMF_VIDEO_EPISODE_INTERVAL=200
export RLVLMF_NUM_TRAIN_STEPS=150000
export RLVLMF_NUM_SEED_STEPS=250
export RLVLMF_NUM_UNSUP_STEPS=2250
export RLVLMF_NUM_INTERACT=1250
export RLVLMF_EVAL_FREQUENCY=2500
export RLVLMF_SAVE_INTERVAL=2000
export RLVLMF_REWARD_BATCH=100
export RLVLMF_REWARD_UPDATE=30
export RLVLMF_REWARD_LR=1e-4
export RLVLMF_MAX_FEEDBACK=20000
export RLVLMF_TERMINATE_ON_SUCCESS=false
export RLVLMF_RESNET=1

cd /project2/biyik_1165/RL-VLM-F_value-model
exec bash scripts/_run_matched_horizon_training.sh
