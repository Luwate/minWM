#!/bin/bash
# run_training_pipeline.sh - Training execution script for minWM on VPS

set -e

# Load conda env if conda command exists
if command -v conda &> /dev/null; then
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate minwm || echo "Warning: Could not activate conda environment 'minwm'."
fi

# Configuration
NUM_GPUS_PER_NODE=${1:-8} # Overrides GPUs (default 8)
STAGE=${2:-"0"}          # Overrides Stage (default 0: Bidirectional SFT)

# Shift positional parameters so "$@" represents any extra arguments (e.g. --disable-wandb)
if [ "$#" -ge 2 ]; then
    shift 2
elif [ "$#" -eq 1 ]; then
    shift 1
fi


# Shift positional parameters so "$@" represents any extra arguments (e.g. --disable-wandb)
if [ "$#" -ge 2 ]; then
    shift 2
elif [ "$#" -eq 1 ]; then
    shift 1
fi


# Dynamically set sequence parallel size based on GPU count
if [ "$NUM_GPUS_PER_NODE" -ge 8 ]; then
    SP_SIZE=4
elif [ "$NUM_GPUS_PER_NODE" -ge 4 ]; then
    SP_SIZE=4
elif [ "$NUM_GPUS_PER_NODE" -ge 2 ]; then
    SP_SIZE=2
else
    SP_SIZE=1
fi

echo "=== minWM Training Pipeline ==="
echo "  Selected Stage: $STAGE"
echo "  Available GPUs: $NUM_GPUS_PER_NODE"
echo "  Sequence Parallel Size (sp_size): $SP_SIZE"
echo "==============================="

case "$STAGE" in
    0)
        echo "--> Running Phase 1: Bidirectional SFT (Camera Control)"
        torchrun \
            --nproc_per_node="$NUM_GPUS_PER_NODE" \
            Wan21/wan_train.py \
            --config_path Wan21/configs/bidirectional_camera.yaml \
            --logdir logs/bidirectional_camera \
            --sp_size "$SP_SIZE" \
            "$@"
        ;;
    1)
        echo "--> Running Phase 2 Stage 1: Teacher Forcing AR Diffusion"
        # Download Phase 1 checkpoint if not present
        if [ ! -f "./ckpts/Wan21/Action2V/bidirectional/model.pt" ]; then
            echo "Phase 1 checkpoint not found. Downloading..."
            huggingface-cli download MIN-Lab/minWM --local-dir ./ckpts --include "Wan21/Action2V/bidirectional/**"
        fi
        torchrun \
            --nproc_per_node="$NUM_GPUS_PER_NODE" \
            Wan21/wan_train.py \
            --config_path Wan21/configs/ar_camera_tf.yaml \
            --logdir logs/ar_camera_tf \
            --sp_size "$SP_SIZE" \
            --tf \
            "$@"
        ;;
    2a)
        echo "--> Running Phase 2 Stage 2a: Causal ODE Distillation"
        # Download Stage 1 checkpoint if not present
        if [ ! -f "./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt" ]; then
            echo "Stage 1 checkpoint not found. Downloading..."
            huggingface-cli download MIN-Lab/minWM --local-dir ./ckpts --include "Wan21/Action2V/ar_diffusion_tf/**"
        fi
        torchrun \
            --nproc_per_node="$NUM_GPUS_PER_NODE" \
            Wan21/wan_train.py \
            --config_path Wan21/configs/causal_ode_camera.yaml \
            --logdir logs/causal_ode_camera \
            --sp_size "$SP_SIZE" \
            "$@"
        ;;
    2b)
        echo "--> Running Phase 2 Stage 2b: Causal Consistency Distillation (CD)"
        # Download Stage 1 checkpoint if not present
        if [ ! -f "./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt" ]; then
            echo "Stage 1 checkpoint not found. Downloading..."
            huggingface-cli download MIN-Lab/minWM --local-dir ./ckpts --include "Wan21/Action2V/ar_diffusion_tf/**"
        fi
        torchrun \
            --nproc_per_node="$NUM_GPUS_PER_NODE" \
            Wan21/wan_train.py \
            --config_path Wan21/configs/causal_cd_camera.yaml \
            --logdir logs/causal_cd_camera \
            --sp_size "$SP_SIZE" \
            "$@"
        ;;
    3)
        echo "--> Running Phase 2 Stage 3: Asymmetric DMD with Self Rollout"
        # Download Stage 2a checkpoint if not present
        if [ ! -f "./ckpts/Wan21/Action2V/causal_ode/model.pt" ] && [ ! -f "./ckpts/Wan21/Action2V/causal_cd/model.pt" ]; then
            echo "Stage 2 checkpoint not found. Downloading ODE initialization..."
            huggingface-cli download MIN-Lab/minWM --local-dir ./ckpts --include "Wan21/Action2V/causal_ode/**"
        fi
        torchrun \
            --nproc_per_node="$NUM_GPUS_PER_NODE" \
            Wan21/wan_train.py \
            --config_path Wan21/configs/causal_forcing_dmd_camera.yaml \
            --logdir logs/causal_dmd_camera \
            --sp_size "$SP_SIZE" \
            "$@"
        ;;
    *)
        echo "Invalid stage specified. Available stages: 0, 1, 2a, 2b, 3"
        exit 1
        ;;
esac

echo "--> Stage $STAGE completed."
