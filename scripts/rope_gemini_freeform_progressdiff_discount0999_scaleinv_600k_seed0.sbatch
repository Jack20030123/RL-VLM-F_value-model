#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=rope_pd0999si_600k_s0
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
    export PYFLEXROOT=/workspace/RL-VLM-F/softgym/PyFlex
    export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:${LD_LIBRARY_PATH}

    cd /workspace/RL-VLM-F

    python train_PEBBLE_with_video.py \
      online_log_root_dir=/scratch1/haobaizh/rlvlmf_online_logs \
      env=softgym_RopeFlattenEasy \
      reward=learn_from_preference \
      vlm_label=1 vlm=gemini_free_form \
      image_reward=1 segment=1 resnet=1 \
      use_progress_diff_reward=true \
      progress_diff_discount=0.999 \
      progress_diff_reward_scale=1.0 \
      progress_diff_scale_by_inv_one_minus_gamma=true \
      reward_batch=100 reward_update=30 reward_lr=1e-4 \
      num_interact=5000 max_feedback=20000 \
      num_train_steps=600000 \
      num_unsup_steps=9000 \
      num_eval_episodes=1 \
      agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 \
      agent.params.batch_size=512 \
      double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
      diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 \
      gradient_update=1 activation=tanh \
      feed_type=0 teacher_beta=-1 teacher_gamma=1 \
      teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 \
      seed=0 \
      exp_name=progressdiff_discount0999_scaleinv_gemini_freeform_600k_seed0_rope
  '
