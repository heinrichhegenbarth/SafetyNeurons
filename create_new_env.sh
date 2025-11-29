#!/bin/bash
#SBATCH --job-name=create_conda_env
#SBATCH --output=create_conda_env.log
#SBATCH --time=04:00:00      # 1 hour
#SBATCH --cpus-per-task=2
#SBATCH --partition=scavenge

# Load Anaconda module
module load Anaconda3

# Initialize Conda for this shell
eval "$(conda shell.bash hook)"

# Define environment path in hheg_stli
ENV_PATH="/home/hheg_stli/condaenvs/myenv"

# Create environment if it doesn't exist
if [ ! -d "$ENV_PATH" ]; then
    conda create -p $ENV_PATH python=3.11 -y
    echo "Conda environment created at $ENV_PATH"
fi

# Activate the environment
conda activate $ENV_PATH

# List of packages to install in the requested order
PACKAGES=("transformers" "datasets" "torch" "jupyterlab" "pandas" "seaborn" "matplotlib")

# Install each package via pip and print a confirmation message
for pkg in "${PACKAGES[@]}"; do
    pip install --upgrade $pkg
    echo "Package $pkg installed successfully"
done

# Verify that all packages can be imported
python -c "import transformers, datasets, torch, jupyterlab, pandas, seaborn, matplotlib; print('All packages imported successfully')"

echo "jobs done :)"
