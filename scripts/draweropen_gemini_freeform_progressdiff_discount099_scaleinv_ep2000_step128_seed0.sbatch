#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --job-name=draweropen_pd099_ep2k128_s0
#SBATCH --output=logs/%x-%j.out

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
  bash -lc '
    set -e
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

    export PYTHONPATH=/workspace/RL-VLM-F/.pydeps:/workspace/RL-VLM-F/softgym:/workspace/RL-VLM-F/softgym/PyFlex/bindings:/workspace/RL-VLM-F/softgym/PyFlex/bindings/build:/workspace/RL-VLM-F:${PYTHONPATH}

    cd /workspace/RL-VLM-F

    python train_PEBBLE_with_video.py \
      online_log_root_dir=/scratch1/haobaizh/rlvlmf_online_logs \
      env=metaworld_drawer-open-v2 \
      reward=learn_from_preference \
      vlm_label=1 vlm=gemini_free_form \
      image_reward=1 segment=1 \
      teacher_eps_mistake=0 \
      use_progress_diff_reward=true \
      use_smooth_relabel=0 \
      progress_diff_discount=0.99 \
      progress_diff_reward_scale=1.0 \
      progress_diff_scale_by_inv_one_minus_gamma=true \
      feed_type=0 teacher_beta=-1 teacher_gamma=1 teacher_eps_skip=0 teacher_eps_equal=0 \
      terminate_on_success=false \
      use_episode=true \
      max_episode_steps=128 \
      image_replay_capacity_episodes=400 \
      num_train_episodes=2000 \
      num_seed_episodes=2 \
      num_unsup_episodes=18 \
      num_interact_episodes=8 \
      eval_episode_frequency=20 \
      save_episode_interval=200 \
      video_episode_interval=200 \
      video_window_episodes=1 \
      save_env_reward_video_success_only=false \
      num_train_steps=256000 \
      num_seed_steps=256 \
      num_unsup_steps=2304 \
      num_interact=1024 \
      eval_frequency=2560 \
      save_interval=25600 \
      reward_batch=40 reward_update=10 \
      max_feedback=20000 \
      gradient_update=1 activation=tanh \
      num_eval_episodes=1 \
      agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 \
      agent.params.batch_size=512 \
      double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
      diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 \
      seed=0 \
      exp_name=progressdiff_discount099_scaleinv_gemini_freeform_ep2000_step128_seed0_draweropen
  '
