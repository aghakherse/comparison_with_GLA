#!/bin/bash -x
#SBATCH --mem=32G
#SBATCH --qos=high     
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=01:00:00


module load Python3/3.11.2
# activate you virtual environment 
# the next 3 lines install our linear attention library (fastmax_cuda). NOTE: gcc version must be < 13.0.0
cd linear_attention
module load gcc
module load cuda
python setup_fastmax.py install
# the next line install GLA
pip install -U git+https://github.com/sustcsonglin/flash-linear-attention

python profiling.py
