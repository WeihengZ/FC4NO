#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16     # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4
#SBATCH --time=12:00:00
#SBATCH --account=bbqg-delta-gpu
#SBATCH --job-name=driver_multi
#SBATCH --output="exps/multi/driver_multi.out"

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
            --bind /taiga/illinois/eng/cee/meidani/Vincent:/taiga/illinois/eng/cee/meidani/Vincent \
            --bind /u/wzhong/PhD/FairCompare:/FC \
            modulus.sif\
    bash -c "cd /FC && echo 'job is starting on \$(hostname)' && python --version && python mainVT.py --model='multi' --data='driver'"
