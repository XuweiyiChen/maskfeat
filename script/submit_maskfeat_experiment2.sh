#!/bin/bash
#SBATCH -A uva_cv_lab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000
#SBATCH --ntasks=1
#SBATCH --time=32:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xuweic@email.virginia.edu
#SBATCH --output=/project/uva_cv_lab/xuweic/SlowFast/script/logs/0_1.log
#SBATCH --error=/project/uva_cv_lab/xuweic/SlowFast/script/logs/0_1.err

module load gcc/11.4.0
module load cuda/12.4.1
cd /project/uva_cv_lab/xuweic/SlowFast

export OMP_NUM_THREADS=1
export TORCH_SHOW_CPP_STACKTRACES=1

# Pretraining
python tools/run_net.py --cfg configs/masked_ssl/in100_VIT_B_MaskFeat_PT_0_1.yaml