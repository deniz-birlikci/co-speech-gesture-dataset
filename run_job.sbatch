#!/bin/bash
#SBATCH -p cpu_long
#SBATCH --pty
#SBATCH --mem=94000
#SBATCH -c 32

# Load any required modules
# module load python

# Activate your virtual environment if necessary
conda activate gesture

# Change to the directory where your Python code is located
# cd /path/to/your/python/code/directory

# Run your Python code
python video_landmarker.py

# Deactivate the virtual environment if necessary
conda deactivate
