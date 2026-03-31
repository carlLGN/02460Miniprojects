#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J ensemble
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### we want to have this on a single node
#BSUB -R "span[hosts=1]"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 12:00
### request 3GB of system-memory
#BSUB -R "rusage[mem=2GB]"
### -- Specify the output and error file. %J is the job-id --
#BSUB -o /work3/s214624/02460Miniprojects/bash_outputs/ensemble_%J.out
#BSUB -e /work3/s214624/02460Miniprojects/bash_outputs/ensemble_%J.err
# -- end of LSF options --

# Print GPU information
nvidia-smi

# Load required modules
module load python3/3.12.4
module load cuda/12.4.1

# Activate virtual environment
source /work3/s214624/02460Miniprojects/.venv/bin/activate

cd /work3/s214624/02460Miniprojects

# Run script
echo "Running script"

uv run python -m Project2.src.ensemble.train_ensemble geodesics


echo "Job completed"

# to run script do: bsub < {path to bashscript}