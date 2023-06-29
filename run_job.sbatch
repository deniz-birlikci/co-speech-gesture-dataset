#!/bin/bash
#SBATCH -p cpu_long
#SBATCH -c 32
#SBATCH --mem=60000

# Load any required modules
# module load python

# Activate your virtual environment if necessary
conda activate gesture

# Change to the directory where your Python code is located
cd /work/mbirlikc/co-speech-gesture-dataset

# Run your Python code
python video_landmarker.py

# Deactivate the virtual environment if necessary
conda deactivate
