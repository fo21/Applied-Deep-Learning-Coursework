#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --job-name=model-training
#SBATCH --account=COMS033444

cd "${SLURM_SUBMIT_DIR}"


module load cuda/12.4.0

echo "Start"

python Mr-CNN.py --epochs 5 

echo "Done"
