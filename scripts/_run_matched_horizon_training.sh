#!/bin/bash
set -euo pipefail

: "${RLVLMF_TASK_ENV:?}"
: "${RLVLMF_METHOD:?}"
: "${RLVLMF_EXP_NAME:?}"
: "${RLVLMF_USE_EPISODE:=true}"
export RLVLMF_USE_EPISODE
: "${RLVLMF_MAX_EPISODE_STEPS:?}"
if [ "$RLVLMF_USE_EPISODE" = "true" ]; then
  : "${RLVLMF_IMAGE_REPLAY_CAPACITY_EPISODES:?}"
  : "${RLVLMF_NUM_TRAIN_EPISODES:?}"
  : "${RLVLMF_NUM_SEED_EPISODES:?}"
  : "${RLVLMF_NUM_UNSUP_EPISODES:?}"
  : "${RLVLMF_NUM_INTERACT_EPISODES:?}"
  : "${RLVLMF_EVAL_EPISODE_FREQUENCY:?}"
  : "${RLVLMF_SAVE_EPISODE_INTERVAL:?}"
  : "${RLVLMF_VIDEO_EPISODE_INTERVAL:?}"
elif [ "$RLVLMF_USE_EPISODE" = "false" ]; then
  : "${RLVLMF_IMAGE_REPLAY_CAPACITY:?}"
else
  echo "RLVLMF_USE_EPISODE must be true or false, got $RLVLMF_USE_EPISODE" >&2
  exit 2
fi
: "${RLVLMF_NUM_TRAIN_STEPS:?}"
: "${RLVLMF_NUM_SEED_STEPS:?}"
: "${RLVLMF_NUM_UNSUP_STEPS:?}"
: "${RLVLMF_NUM_INTERACT:?}"
: "${RLVLMF_EVAL_FREQUENCY:?}"
: "${RLVLMF_SAVE_INTERVAL:?}"
: "${RLVLMF_REWARD_BATCH:?}"
: "${RLVLMF_REWARD_UPDATE:?}"
: "${RLVLMF_REWARD_LR:?}"
: "${RLVLMF_MAX_FEEDBACK:?}"
: "${RLVLMF_TERMINATE_ON_SUCCESS:?}"
: "${RLVLMF_RESNET:?}"
: "${RLVLMF_USE_SMOOTH_RELABEL:=0}"
: "${RLVLMF_VIDEO_STEP_INTERVAL:=}"
: "${RLVLMF_VIDEO_STEP_OFFSET:=}"
export RLVLMF_USE_SMOOTH_RELABEL RLVLMF_VIDEO_STEP_INTERVAL RLVLMF_VIDEO_STEP_OFFSET

module purge
module load apptainer

export RLVLMF_STORAGE_ROOT=/scratch1/haobaizh
mkdir -p "$RLVLMF_STORAGE_ROOT/apptainer" "$RLVLMF_STORAGE_ROOT/rlvlmf_online_logs"
export APPTAINER_CACHEDIR="$RLVLMF_STORAGE_ROOT/apptainer"

cd /project2/biyik_1165/RL-VLM-F_value-model

apptainer exec --nv \
  -B /project2/biyik_1165/RL-VLM-F_value-model:/workspace/RL-VLM-F \
  -B "$RLVLMF_STORAGE_ROOT":"$RLVLMF_STORAGE_ROOT" \
  rlvlmf_value_model_carc.sif \
  bash -s <<'INNER'
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate rlvlmf

if [ -f /workspace/RL-VLM-F/.env ]; then
  set -a
  source /workspace/RL-VLM-F/.env
  set +a
fi

export HF_HOME=/workspace/RL-VLM-F/hf_cache
export TRANSFORMERS_CACHE=/workspace/RL-VLM-F/hf_cache/transformers
export HUGGINGFACE_HUB_CACHE=/workspace/RL-VLM-F/hf_cache/hub
export TORCH_HOME=/workspace/RL-VLM-F/torch_cache
export TIMM_HOME=/workspace/RL-VLM-F/timm_cache
export XDG_CACHE_HOME=/workspace/RL-VLM-F/.cache
export RLVLMF_ONLINE_LOG_ROOT="${RLVLMF_STORAGE_ROOT}/rlvlmf_online_logs"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE" \
         "$TORCH_HOME" "$TIMM_HOME" "$XDG_CACHE_HOME" \
         "$RLVLMF_ONLINE_LOG_ROOT"

export PYTHONPATH=/workspace/RL-VLM-F/.pydeps:/workspace/RL-VLM-F/softgym:/workspace/RL-VLM-F/softgym/PyFlex/bindings:/workspace/RL-VLM-F/softgym/PyFlex/bindings/build:/workspace/RL-VLM-F:${PYTHONPATH:-}
export PYFLEXROOT=/workspace/RL-VLM-F/softgym/PyFlex
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:${LD_LIBRARY_PATH:-}

cd /workspace/RL-VLM-F

cmd=(
  python train_PEBBLE_with_video.py
  online_log_root_dir=/scratch1/haobaizh/rlvlmf_online_logs
  env="$RLVLMF_TASK_ENV"
  reward=learn_from_preference
  vlm_label=1
  vlm=gemini_free_form
  image_reward=1
  segment=1
  teacher_eps_mistake=0
  use_smooth_relabel="$RLVLMF_USE_SMOOTH_RELABEL"
  feed_type=0
  teacher_beta=-1
  teacher_gamma=1
  teacher_eps_skip=0
  teacher_eps_equal=0
  terminate_on_success="$RLVLMF_TERMINATE_ON_SUCCESS"
  use_episode="$RLVLMF_USE_EPISODE"
  max_episode_steps="$RLVLMF_MAX_EPISODE_STEPS"
  video_window_episodes=1
  save_env_reward_video_success_only=false
  num_train_steps="$RLVLMF_NUM_TRAIN_STEPS"
  num_seed_steps="$RLVLMF_NUM_SEED_STEPS"
  num_unsup_steps="$RLVLMF_NUM_UNSUP_STEPS"
  num_interact="$RLVLMF_NUM_INTERACT"
  eval_frequency="$RLVLMF_EVAL_FREQUENCY"
  save_interval="$RLVLMF_SAVE_INTERVAL"
  reward_batch="$RLVLMF_REWARD_BATCH"
  reward_update="$RLVLMF_REWARD_UPDATE"
  reward_lr="$RLVLMF_REWARD_LR"
  max_feedback="$RLVLMF_MAX_FEEDBACK"
  gradient_update=1
  activation=tanh
  num_eval_episodes=1
  agent.params.actor_lr=0.0003
  agent.params.critic_lr=0.0003
  agent.params.batch_size=512
  double_q_critic.params.hidden_dim=256
  double_q_critic.params.hidden_depth=3
  diag_gaussian_actor.params.hidden_dim=256
  diag_gaussian_actor.params.hidden_depth=3
  seed=0
  exp_name="$RLVLMF_EXP_NAME"
)

if [ "$RLVLMF_USE_EPISODE" = "true" ]; then
  cmd+=(
    image_replay_capacity_episodes="$RLVLMF_IMAGE_REPLAY_CAPACITY_EPISODES"
    num_train_episodes="$RLVLMF_NUM_TRAIN_EPISODES"
    num_seed_episodes="$RLVLMF_NUM_SEED_EPISODES"
    num_unsup_episodes="$RLVLMF_NUM_UNSUP_EPISODES"
    num_interact_episodes="$RLVLMF_NUM_INTERACT_EPISODES"
    eval_episode_frequency="$RLVLMF_EVAL_EPISODE_FREQUENCY"
    save_episode_interval="$RLVLMF_SAVE_EPISODE_INTERVAL"
    video_episode_interval="$RLVLMF_VIDEO_EPISODE_INTERVAL"
  )
else
  cmd+=(image_replay_capacity="$RLVLMF_IMAGE_REPLAY_CAPACITY")
fi

if [ -n "$RLVLMF_VIDEO_STEP_INTERVAL" ]; then
  cmd+=(video_step_interval="$RLVLMF_VIDEO_STEP_INTERVAL")
fi

if [ -n "$RLVLMF_VIDEO_STEP_OFFSET" ]; then
  cmd+=(video_step_offset="$RLVLMF_VIDEO_STEP_OFFSET")
fi

if [ "$RLVLMF_RESNET" = "1" ]; then
  cmd+=(resnet=1)
fi

case "$RLVLMF_METHOD" in
  baseline)
    cmd+=(use_progress_diff_reward=false)
    ;;
  progressdiff0999_scaleinv)
    cmd+=(
      use_progress_diff_reward=true
      progress_diff_discount=0.999
      progress_diff_reward_scale=1.0
      progress_diff_scale_by_inv_one_minus_gamma=true
    )
    ;;
  *)
    echo "Unknown RLVLMF_METHOD=$RLVLMF_METHOD" >&2
    exit 2
    ;;
esac

"${cmd[@]}"
INNER
