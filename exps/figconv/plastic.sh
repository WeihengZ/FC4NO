#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16     # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4
#SBATCH --time=12:00:00
#SBATCH --account=bbqg-delta-gpu
#SBATCH --job-name=plastic_figconv
#SBATCH --output="exps/figconv/plastic_figconv.out"

### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=verbose,per_task:1
###SBATCH --gpu-bind=none     # <- or closest

module purge # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)

module load openmpi
source ~/.bashrc  # Ensure conda is set up

conda deactivate
cd /work/hdd/bbqg/wzhong/container

apptainer run --nv --writable-tmpfs \
    --no-home \
    --bind /projects:/projects \
    --bind /work/hdd/bbqg:/work/hdd/bbqg \
    --bind /work/hdd/bdsy:/work/hdd/bdsy \
    --bind /work/nvme/bdsy:/work/nvme/bdsy \
    --bind /u/wzhong/PhD/FairCompare:/FC \
    modulus.sif \
    bash -c "cd /FC && echo 'job is starting on \$(hostname)' && python --version && python main.py --model='figconv' --data='plastic'"
