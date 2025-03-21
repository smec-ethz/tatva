#!/bin/bash

#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8096
#SBATCH --time=01:00:00
#SBATCH --job-name=test-bar
#SBATCH --output=/cluster/home/mpundir/studies/lattice-damage-model/simulations/analysis.out
#SBATCH --error=/cluster/home/mpundir/studies/lattice-damage-model/simulations/analysis.err

source /cluster/project/cmbm/local-stacks/load-scripts/load_gpu.sh 
source /cluster/work/cmbm/mpundir/venv/my-venv/bin/activate
export JAX_CACHE_DIR="$SCRATCH/jax-cache"
export JAX_PLATFORM="cpu"
python /cluster/home/mpundir/studies/lattice-damage-model/simulations/bar.py 
