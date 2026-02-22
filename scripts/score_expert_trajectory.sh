#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --job-name=score_expert_trajectory
#SBATCH --output=logs/%x-%j.out

# ─── User-configurable arguments ──────────────────────────────────────────────
ENV_NAME="metaworld_drawer-open-v2"
MODEL_DIR="models_gt"
STEP=500000
REWARD_MODEL_DIR="models"
REWARD_MODEL_STEP=1000000
ENSEMBLE_SIZE=3
SEED=0
MAX_STEPS=500
IMAGE_SIZE=300
OUTPUT_DIR="expert_trajectory_output"
# ──────────────────────────────────────────────────────────────────────────────

module purge
module load apptainer

export APPTAINER_CACHEDIR=/scratch1/haobaizh/apptainer
mkdir -p "$APPTAINER_CACHEDIR"
mkdir -p logs

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

    # Headless rendering for MuJoCo on cluster
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    export MPLBACKEND=Agg

    export PYTHONPATH=/workspace/RL-VLM-F/.pydeps:/workspace/RL-VLM-F/softgym:/workspace/RL-VLM-F/softgym/PyFlex/bindings:/workspace/RL-VLM-F/softgym/PyFlex/bindings/build:/workspace/RL-VLM-F:${PYTHONPATH}

    cd /workspace/RL-VLM-F

    python score_expert_trajectory.py \
      --env            "'"$ENV_NAME"'" \
      --model_dir      "'"$MODEL_DIR"'" \
      --step           '"$STEP"' \
      --reward_model_dir "'"$REWARD_MODEL_DIR"'" \
      --reward_model_step '"$REWARD_MODEL_STEP"' \
      --ensemble_size  '"$ENSEMBLE_SIZE"' \
      --seed           '"$SEED"' \
      --max_steps      '"$MAX_STEPS"' \
      --image_size     '"$IMAGE_SIZE"' \
      --output_dir     "'"$OUTPUT_DIR"'"
  '
