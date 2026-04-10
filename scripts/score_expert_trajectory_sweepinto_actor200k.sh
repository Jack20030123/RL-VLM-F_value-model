#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=score_sweepinto_a200k
#SBATCH --output=logs/%x-%j.out

ACTOR_STEP=200000
ENSEMBLE_SIZE=3
SEED=0
MAX_STEPS=500
IMAGE_SIZE=300
OUTPUT_ROOT="expert_trajectory_output"

module purge
module load apptainer

export APPTAINER_CACHEDIR=/scratch1/haobaizh/apptainer
mkdir -p "$APPTAINER_CACHEDIR"
mkdir -p logs

cd /project2/biyik_1165/RL-VLM-F_value-model

apptainer exec --nv \
  -B /project2/biyik_1165/RL-VLM-F_value-model:/workspace/RL-VLM-F \
  -B /scratch1:/scratch1 \
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

    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    export MPLBACKEND=Agg

    export PYTHONPATH=/workspace/RL-VLM-F/.pydeps:/workspace/RL-VLM-F/softgym:/workspace/RL-VLM-F/softgym/PyFlex/bindings:/workspace/RL-VLM-F/softgym/PyFlex/bindings/build:/workspace/RL-VLM-F:${PYTHONPATH}

    cd /workspace/RL-VLM-F

    python score_expert_trajectory.py \
      --env "metaworld_sweep-into-v2" \
      --model_dir "/scratch1/haobaizh/rlvlmf_online_logs/gt_reward_sweepinto/metaworld_sweep-into-v2/2026-04-07-03-56-43/vlm_0bard_rewardgt_task_reward_H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/label_smooth_0.0/schedule_0/PEBBLE_init1000_unsup9000_inter5000_maxfeed20000_seg50_acttanh_Rlr0.0001_Rbatch100_Rupdate30_en3_sample0_large_batch10_seed0/models" \
      --step '"$ACTOR_STEP"' \
      --reward_model_dir "/scratch1/haobaizh/rlvlmf_online_logs/baseline_gemini_freeform_1M_noterminate_sweepinto/metaworld_sweep-into-v2/2026-03-02-08-21-42/vlm_1gemini_free_form_rewardlearn_from_preference_H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/label_smooth_0.0/schedule_0/PEBBLE_init1000_unsup9000_inter4000_maxfeed20000_seg1_acttanh_Rlr0.0003_Rbatch40_Rupdate10_en3_sample0_large_batch10_seed1/models" \
      --reward_model_step 600000 \
      --ensemble_size '"$ENSEMBLE_SIZE"' \
      --seed '"$SEED"' \
      --max_steps '"$MAX_STEPS"' \
      --image_size '"$IMAGE_SIZE"' \
      --output_dir "'"$OUTPUT_ROOT"'/sweepinto_actor200k"
  '
