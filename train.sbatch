#!/bin/bash
#SBATCH --job-name=train
#SBATCH -p gpu-preempt
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eou pipefail

# start in submission directory
cd "${SLURM_SUBMIT_DIR}"

# ensure log dir exists
mkdir -p logs

# load conda and activate env
module purge
module load conda/latest
conda activate asn4

# single-process launch
python TransformerPerceiverAR.py