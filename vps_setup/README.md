# 🚀 minWM VPS Setup & Training Guide

This guide details how to configure, run, and experiment with the **minWM** full-stack real-time interactive world model framework on a Virtual Private Server (VPS).

---

## 💻 1. VPS Hardware Recommendations

Before starting, ensure your VPS meets the minimum requirements for the selected backbone model:

### Option A: Wan 2.1 (1.3B Parameters) - *Recommended*
* **GPU:** Minimum 1x A100 (40GB/80GB) or 1x H100. For multi-GPU sequence parallelism, 2x or 4x L4 (24GB) or A10G (24GB) can also be used.
* **Storage:** 150GB+ SSD/NVMe (to hold model weights, video datasets, and LMDB files).
* **RAM:** 60GB+ system RAM.

### Option B: HunyuanVideo 1.5 (8B Parameters)
* **GPU:** Minimum 4x A100 (80GB) or 4x H100 to support sequence parallelism.
* **Storage:** 300GB+ SSD/NVMe.
* **RAM:** 120GB+ system RAM.

---

## 🛠️ 2. Step-by-Step Environment Provisioning

### Step 2.1: Clone the Repository to the VPS
Connect to your VPS and clone the repository:
```bash
git clone git@github.com:Luwate/minWM.git
cd minWM
```

### Step 2.2: Automated Dependency Installation
Run the helper script `vps_setup/install_deps.sh` to initialize the environment:
```bash
bash vps_setup/install_deps.sh
```
This script will:
1. Create a Conda environment named `minwm` (Python 3.10).
2. Upgrade `pip` and install all required python packages from `requirements.txt`.
3. Compile and install `flash-attn` without build isolation.
4. Auto-inject the correct `PYTHONPATH` (`HY15`, `Wan21`, `shared`) to load upon activating the conda environment.

Activate the environment:
```bash
conda activate minwm
```

---

## 📦 3. Model Weight Downloads

All weights must land under `./ckpts/` directory.

### Download Wan 2.1 Base Model (required for Wan Action2V)
```bash
# 1. Download base weights from Hugging Face
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./ckpts/Wan2.1-T2V-1.3B

# 2. Create the symlink required by the Wan scripts
mkdir -p Wan21/wan_models
ln -s "$(realpath ./ckpts/Wan2.1-T2V-1.3B)" Wan21/wan_models/Wan2.1-T2V-1.3B
```

### Download HunyuanVideo 1.5 Base Model & Encoders (required for HY Action2V / TI2V)
```bash
# 1. Download HunyuanVideo VAE & Scheduler
huggingface-cli download tencent/HunyuanVideo-1.5 --local-dir ./ckpts/HunyuanVideo-1.5 \
    --include "vae/*" "scheduler/*" "transformer/480p_i2v/*"

# 2. Download Text Encoders (Qwen2.5-VL and Byt5)
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./ckpts/HunyuanVideo-1.5/text_encoder/llm
huggingface-cli download google/byt5-small --local-dir ./ckpts/HunyuanVideo-1.5/text_encoder/byt5-small

# 3. Download Vision Encoder (SigLIP)
huggingface-cli download black-forest-labs/FLUX.1-Redux-dev --local-dir ./ckpts/HunyuanVideo-1.5/vision_encoder/siglip
```

---

## 🗃️ 4. Data Preparation & Pre-encoding

You can choose to use the standard toy dataset (RealEstate10k subset) or import your custom videos.

### Dataset File Layout Structure
Regardless of the data source, the final directory structure should look like this:
```
./dataset/
├── preencode_input.json
├── videos/
│   ├── 000000_right8a11/gen.mp4
│   ├── 000001_w10d9/gen.mp4
│   └── ...
```

#### Custom Data Format:
1. **`preencode_input.json`**: A JSON list where each element specifies an image, caption, and camera pose trajectory string.
   ```json
   [
       {
           "image_path": "./dataset/videos/000000_right8a11/frame_0.png",
           "caption": "A living room scene, panning right",
           "pose_str": "right-8, a-11"
       }
   ]
   ```
2. **`videos/`**: Contains subfolders named `{idx:06d}_{slug(pose_str)}/` containing the corresponding `gen.mp4` video.

### Run Pre-encoding Pipeline (Option 1: Automated Script)
```bash
# Run the dataset prep helper. Pass the number of GPUs you want to utilize (e.g. 4)
bash vps_setup/prepare_dataset.sh 4
```
This script automates:
1. Downloading the `MIN-Lab/minWM-data` toy dataset if no local data is found.
2. Invoking `build_worldplaygen_lmdb.py` to compress and pre-encode raw videos into a merged LMDB format stored at `./dataset/Wan21/Action2V/data/`.
3. Offering a menu to download/build the Phase 2 ODE trajectory database (`./dataset/Wan21/Action2V/ode_lmdb/`).

---

## 🚀 5. Running the Multi-stage Training

Training is divided into two phases. You can run individual stages using `vps_setup/run_training_pipeline.sh <num_gpus> <stage>`:

```bash
# Example: Run Stage 0 (Bidirectional SFT) on 4 GPUs
bash vps_setup/run_training_pipeline.sh 4 0
```

### Training Stage Overview

#### Phase 1: Bidirectional SFT (Stage `0`)
Adapt the bidirectional video diffusion model (T2V) to support camera-control parameters injected via PRoPE.
* **Trainer class:** `camera_bidirectional_diffusion`
* **Output Checkpoints:** `logs/bidirectional_camera/checkpoint_model_XXXXXX/model.pt`

#### Phase 2 Stage 1: Teacher Forcing AR Diffusion (Stage `1`)
Converts the bidirectional model into an autoregressive model using causal masking.
* **Trainer class:** `CausalWanModel` (teacher-forced)
* **Output Checkpoints:** `logs/ar_camera_tf/checkpoint_model_XXXXXX/model.pt`

#### Phase 2 Stage 2a: Causal ODE Distillation (Stage `2a`)
Distills the autoregressive generator using probability flow ODE trajectories.
* **Trainer class:** `camera_ode`
* **Input Dataset:** Compiled ODE LMDB (`./dataset/Wan21/Action2V/ode_lmdb/`)
* **Output Checkpoints:** `logs/causal_ode_camera/checkpoint_model_XXXXXX/model.pt`

#### Phase 2 Stage 2b: Causal Consistency Distillation (Stage `2b`)
*Alternately/Additionally:* Distills the model using Causal Consistency Distillation (Causal CD from Causal Forcing++) where the student maps outputs consistently between adjacent timesteps.
* **Trainer class:** `camera_consistency_distillation`
* **Output Checkpoints:** `logs/causal_cd_camera/checkpoint_model_XXXXXX/model.pt`

#### Phase 2 Stage 3: Asymmetric DMD with Self Rollout (Stage `3`)
Scores and refines the student model on its own autoregressive rollouts using score distillation matching, eliminating error drift.
* **Trainer class:** `camera_score_distillation`
* **Output Checkpoints:** `logs/causal_dmd_camera/checkpoint_model_XXXXXX/model.pt`

---

## 🎛️ 6. Scaling Configuration & Sequences Parallelism

Sequence Parallelism (`--sp_size`) partitions video sequences across GPUs along the temporal dimension to fit larger sequence lengths and save memory.

* **Single GPU:** Set `--sp_size 1` in the training execution command.
* **Multi-GPU (e.g. 4 GPUs):** Set `--sp_size 4` (or `--sp_size 2` depending on VRAM usage vs throughput balance).

Ensure `NUM_GPUS_PER_NODE` is divisible by `SP_SIZE`.

---

## 🕹️ 7. Live Inference & Validation

Verify model predictions on arbitrary camera paths:

### Wan 2.1 Action2V Inference
Run rollout inference using camera key strings:
```bash
CHECKPOINT_PATH="./ckpts/Wan21/Action2V/dmd/model.pt" \
OUTPUT_FOLDER="./outputs/eval_dmd_wan" \
TRAJECTORY_PATH="Wan21/prompts/trajectories.txt" \
    bash Wan21/scripts/inference/run_infer_causal_camera.sh
```

* **Interactive Control:** Adjust the paths in `Wan21/prompts/trajectories.txt` (using keys `w`/`a`/`s`/`d` representing forward/left/backward/right movements, repeated using `*N` notation).

---

## 📊 8. Monitoring Training

By default, training scripts log statistics via **Weights & Biases (wandb)**. 
To track metrics live on your local machine:
1. Ensure your VPS has network access to `wandb.ai`.
2. Configure your credentials in `Wan21/configs/*.yaml`:
   ```yaml
   wandb_key: <your_wandb_key>
   wandb_entity: <your_wandb_entity>
   ```
3. Alternatively, pass `--disable-wandb` to `wan_train.py` to only log metrics to local terminal standard outputs and TensorBoard files under the `--logdir` folder.
