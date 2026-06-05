#!/bin/bash
# prepare_dataset.sh - Dataset preparation script for minWM on VPS

set -e

# Load conda env if conda command exists
if command -v conda &> /dev/null; then
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate minwm || echo "Warning: Could not activate conda environment 'minwm'. Running with current environment."
fi

# Ensure PYTHONPATH includes Wan21/shared
export PYTHONPATH="$PWD/Wan21:$PWD/shared:$PYTHONPATH"

# Configuration
VAE_PATH="Wan21/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
INPUT_JSON="./dataset/preencode_input.json"
VIDEO_DIR="./dataset/videos"
OUTPUT_DIR="./dataset/Wan21/Action2V"
NUM_GPUS_PER_NODE=${1:-8} # Allow overriding GPU count (default 8)

echo "=== minWM Dataset Preparation ==="
echo "Using GPUs: $NUM_GPUS_PER_NODE"
echo "================================="

# Step 1: Check/Download VAE Base Model
if [ ! -f "$VAE_PATH" ]; then
    echo "Base VAE checkpoint not found at $VAE_PATH. Downloading Wan2.1 base models..."
    mkdir -p ./ckpts/Wan2.1-T2V-1.3B
    huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./ckpts/Wan2.1-T2V-1.3B 

    # Create the required symlink
    mkdir -p Wan21/wan_models
    if [ ! -e "Wan21/wan_models/Wan2.1-T2V-1.3B" ]; then
        ln -s "$(realpath ./ckpts/Wan2.1-T2V-1.3B)" Wan21/wan_models/Wan2.1-T2V-1.3B
    fi
else
    echo "Found VAE checkpoint at $VAE_PATH."
fi

# Step 2: Download or Setup Dataset
if [ ! -f "$INPUT_JSON" ]; then
    echo "preencode_input.json not found. Downloading the minWM toy dataset (RealEstate10k subset)..."
    mkdir -p ./dataset
    huggingface-cli download MIN-Lab/minWM-data --repo-type dataset \
        --local-dir ./dataset \
        --include "preencode_input.json" "videos/**"
else
    echo "Dataset JSON found at $INPUT_JSON."
fi

# Step 3: Run Phase 1 SFT LMDB Pre-encoding
echo "=== Phase 1: Encoding raw videos into merged LMDB ==="
if [ -d "$OUTPUT_DIR/data" ]; then
    echo "Output LMDB directory already exists at $OUTPUT_DIR/data. Skipping pre-encoding."
else
    echo "Running build_worldplaygen_lmdb.py using torchrun..."
    torchrun \
        --nproc_per_node="$NUM_GPUS_PER_NODE" \
        Wan21/scripts/data_preprocessing/build_worldplaygen_lmdb.py \
        --input_json "$INPUT_JSON" \
        --video_dir "$VIDEO_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --vae_path "$VAE_PATH"
    echo "Phase 1 LMDB generated at $OUTPUT_DIR/data."
fi

# Step 4: Run Phase 2 Stage 2(a) ODE Distillation Data Prep
echo "=== Phase 2: Preparing ODE Trajectory Data ==="
if [ -d "$OUTPUT_DIR/ode_lmdb" ]; then
    echo "ODE LMDB already exists at $OUTPUT_DIR/ode_lmdb. Skipping."
else
    echo "Do you want to download pre-generated ODE latents (Option A) or generate them from Stage 1 ckpt (Option B)?"
    echo "Select option [A/B]:"
    read -r opt
    if [[ "$opt" == "A" || "$opt" == "a" ]]; then
        echo "Option A: Downloading pre-generated ODE latents from Hugging Face..."
        huggingface-cli download MIN-Lab/minWM-data --repo-type dataset \
            --local-dir ./dataset \
            --include "ODE_data/Wan21/Action2V/**"
        
        echo "Merging downloaded latents into ODE LMDB..."
        python Wan21/wan_utils/build_ode_prope_lmdb.py \
            --input_dir ./dataset/ODE_data/Wan21/Action2V \
            --output_dir "$OUTPUT_DIR/ode_lmdb" \
            --map_size_gb 1000
    else
        echo "Option B: Generating ODE latents from Stage 1 checkpoint..."
        if [ ! -f "./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt" ]; then
            echo "Error: Stage 1 checkpoint not found at ./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt."
            echo "Please download it or train it first before selecting Option B."
            exit 1
        fi
        torchrun --nproc_per_node="$NUM_GPUS_PER_NODE" Wan21/get_causal_ode_data_prope.py \
            --generator_ckpt ./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt \
            --rawdata_path "$OUTPUT_DIR/data" \
            --output_folder "$OUTPUT_DIR/ode_latents"
        
        echo "Merging generated latents into ODE LMDB..."
        python Wan21/wan_utils/build_ode_prope_lmdb.py \
            --input_dir "$OUTPUT_DIR/ode_latents" \
            --output_dir "$OUTPUT_DIR/ode_lmdb" \
            --map_size_gb 1000
    fi
    echo "ODE LMDB compilation completed at $OUTPUT_DIR/ode_lmdb."
fi

echo "=== Dataset Preparation Completed! ==="
