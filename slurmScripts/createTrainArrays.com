#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=i.mawby1@lancaster.ac.uk

source /etc/profile

echo 'BEGIN'
echo Job running on compute node `uname -n`

cd /home/hpc/30/mawbyi1/Ivysaurus

module add cuda
module add anaconda3-gpu
source activate opence_env

python -m pip install uproot
python CreateTrainArrays.py

echo 'DONE'
