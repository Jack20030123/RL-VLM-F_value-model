#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name=draweropen_gemini_freeform_progressdiff
#SBATCH --output=logs/%x-%j.out

module purge
module load apptainer

export APPTAINER_CACHEDIR=/scratch1/haobaizh/apptainer
mkdir -p "$APPTAINER_CACHEDIR"

cd /project2/biyik_1165/RL-VLM-F_value-model

apptainer exec --nv \
  -B /project2/biyik_1165/RL-VLM-F_value-model:/workspace/RL-VLM-F \
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
    mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE" \
             "$TORCH_HOME" "$TIMM_HOME" "$XDG_CACHE_HOME"

    export PYTHONPATH=/workspace/RL-VLM-F/.pydeps:/workspace/RL-VLM-F/softgym:/workspace/RL-VLM-F/softgym/PyFlex/bindings:/workspace/RL-VLM-F/softgym/PyFlex/bindings/build:/workspace/RL-VLM-F:${PYTHONPATH}

    cd /workspace/RL-VLM-F

    python train_PEBBLE_with_video.py \
      env=metaworld_drawer-open-v2 \
      reward=learn_from_preference \
      vlm_label=1 vlm=gemini_free_form \
      image_reward=1 segment=1 \
      use_progress_diff_reward=true \
      image_replay_capacity=200000 \
      num_interact=4000 \
      reward_batch=40 \
      max_feedback=20000 \
      num_train_steps=100001 \
      eval_frequency=5000 num_eval_episodes=5 save_interval=5000 \
      exp_name=progress_diff_gemini_freeform_100k_draweropen
  '
