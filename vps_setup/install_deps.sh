#!/bin/bash
# install_deps.sh - VPS Dependency Installation Script for minWM

set -e

ENV_NAME="minwm"
PYTHON_VERSION="3.10"

echo "=== 1. Creating Conda Environment: $ENV_NAME (Python $PYTHON_VERSION) ==="
if conda info --envs | grep -q "^$ENV_NAME "; then
    echo "Conda environment '$ENV_NAME' already exists. Activating..."
else
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# Note: We must activate the environment in the subshell.
# Since we are in a bash script, we source the conda profile.
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "=== 2. Upgrading pip and installing requirements.txt ==="
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Error: requirements.txt not found in current directory!"
    exit 1
fi

echo "=== 3. Installing flash-attn (no-build-isolation) ==="
echo "Note: This can take some time as it compiles CUDA kernels."
pip install flash-attn --no-build-isolation

echo "=== 4. Setting environment environment variables ==="
# We append these to the conda environment activation script so they load automatically
ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$ACTIVATE_DIR"

cat << 'EOF' > "$ACTIVATE_DIR/minwm_env.sh"
#!/bin/bash
export PROJECT_ROOT="$(pwd)"
export PYTHONPATH="$PROJECT_ROOT/HY15:$PROJECT_ROOT/Wan21:$PROJECT_ROOT/shared:$PYTHONPATH"
echo "minWM Environment activated! PYTHONPATH set."
EOF

echo "=== Setup Completed Successfully ==="
echo "To activate this environment in your shell, run:"
echo "  conda activate $ENV_NAME"
