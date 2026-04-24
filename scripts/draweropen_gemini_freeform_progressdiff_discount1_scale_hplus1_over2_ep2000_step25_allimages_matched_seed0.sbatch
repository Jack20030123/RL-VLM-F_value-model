#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=52G
#SBATCH --time=48:00:00
#SBATCH --job-name=drawerop_pd1shp1o2_h25_s0
#SBATCH --output=logs/%x-%j.out

export RLVLMF_TASK_ENV=metaworld_drawer-open-v2
export RLVLMF_METHOD=progressdiff_discount1_scale_hplus1_over2
export RLVLMF_EXP_NAME=progressdiff_discount1_scale_hplus1_over2_gemini_freeform_ep2000_step25_allimages_matched_seed0_draweropen
export RLVLMF_USE_EPISODE=true
export RLVLMF_MAX_EPISODE_STEPS=25
export RLVLMF_IMAGE_REPLAY_CAPACITY_EPISODES=2000
export RLVLMF_NUM_TRAIN_EPISODES=2000
export RLVLMF_NUM_SEED_EPISODES=2
export RLVLMF_NUM_UNSUP_EPISODES=18
export RLVLMF_NUM_INTERACT_EPISODES=8
export RLVLMF_EVAL_EPISODE_FREQUENCY=20
export RLVLMF_SAVE_EPISODE_INTERVAL=200
export RLVLMF_VIDEO_EPISODE_INTERVAL=200
export RLVLMF_NUM_TRAIN_STEPS=50000
export RLVLMF_NUM_SEED_STEPS=50
export RLVLMF_NUM_UNSUP_STEPS=450
export RLVLMF_NUM_INTERACT=200
export RLVLMF_EVAL_FREQUENCY=500
export RLVLMF_SAVE_INTERVAL=5000
export RLVLMF_REWARD_BATCH=40
export RLVLMF_REWARD_UPDATE=10
export RLVLMF_REWARD_LR=0.0003
export RLVLMF_MAX_FEEDBACK=20000
export RLVLMF_TERMINATE_ON_SUCCESS=false
export RLVLMF_RESNET=0

cd /project2/biyik_1165/RL-VLM-F_value-model
exec bash scripts/_run_matched_horizon_training.sh
