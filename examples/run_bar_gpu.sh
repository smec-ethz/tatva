#!/bin/bash

#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100-pcie-40gb:1
#SBATCH --gres=gpumem:20g
#SBATCH --time=01:00:00
#SBATCH --job-name=jax-gpu
#SBATCH --output=/cluster/home/mpundir/studies/lattice-damage-model/simulations/analysis-gpu.out
#SBATCH --error=/cluster/home/mpundir/studies/lattice-damage-model/simulations/analysis-gpu.err

source /cluster/project/cmbm/local-stacks/load-scripts/load_gpu.sh 
source /cluster/work/cmbm/mpundir/venv/my-venv/bin/activate
export JAX_CACHE_DIR="$SCRATCH/jax-cache"
export JAX_PLATFORM="gpu"
python /cluster/home/mpundir/studies/lattice-damage-model/simulations/bar.py