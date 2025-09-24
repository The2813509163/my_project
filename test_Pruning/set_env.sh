#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Common Variables ---
PYTHON_VERSION="3.10"
TORCH_COMMAND="pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124"

# --- Function to create and setup an environment ---
# $1: Environment name
# $2: Comma-separated list of pip packages to install (beyond torch)
setup_env() {
    ENV_NAME=$1
    shift # Remove the first argument (env name)
    PIP_PACKAGES_TO_INSTALL=("$@") # Remaining arguments are pip packages

    echo "======================================================================"
    echo "Creating and setting up Conda environment: ${ENV_NAME}"
    echo "======================================================================"

    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

    echo "--- Installing PyTorch in ${ENV_NAME} ---"
    conda run -n "${ENV_NAME}" ${TORCH_COMMAND}

    if [ ${#PIP_PACKAGES_TO_INSTALL[@]} -gt 0 ]; then
        echo "--- Installing additional packages in ${ENV_NAME} ---"
        # Combine all pip install commands into one for efficiency
        INSTALL_CMD="pip install"
        for pkg in "${PIP_PACKAGES_TO_INSTALL[@]}"; do
            INSTALL_CMD="${INSTALL_CMD} ${pkg}"
        done
        conda run -n "${ENV_NAME}" ${INSTALL_CMD}
    fi

    echo "--- Environment ${ENV_NAME} setup complete. ---"
    echo
}

# --- Main Script ---

echo "Starting environment setup script..."
echo "This script assumes that directories like 'thirdparty/', 'LLaMA-Factory/', etc., are present in the current working directory if installing with '-e'."
echo "Make sure you are in the correct parent directory before running."
echo

# 1. PAT 定制 llama-factory
setup_env "factory-pat" \
    "-e thirdparty/transformers-4.51.1" \
    "-e thirdparty/peft-0.15.1" \
    "tensorboard" \
    "-e LLaMA-Factory"

# 2. PAT 定制 opencompass
setup_env "opencompass-pat" \
    "opencompass" \
    "-e thirdparty/transformers-4.51.1" \
    "-e thirdparty/peft-0.15.1"

# 3. 官方 llama-factory
setup_env "factory-official" \
    "transformers==4.51.1" \
    "peft==0.15.1" \
    "tensorboard" \
    "-e LLaMA-Factory-main"

# 4. 官方 opencompass
setup_env "opencompass-official" \
    "opencompass" \
    "transformers==4.51.1" \
    "peft==0.15.1"

echo "======================================================================"
echo "All environments have been processed."
echo "To use an environment, run: conda activate <environment_name>"
echo "For example: conda activate factory-pat"
echo "======================================================================"