#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
# #SBATCH --partition=devel
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1        # 1 gpus per node out of 4
#SBATCH --mem=64GB
#SBATCH --nodelist=lnode13

#if [ -z "${2}" ]; then
#  echo usage: sbatch ${0} executable  outfile
#  exit 1
#fi


module load Core/lmod/6.6
module load Core/settarg/6.6

export OMP_PROC_BIND=true 
export OMP_PLACES=cores

srun  ./main -i example.in &> output_GPU
#srun  ./main -i example.in &> output_GPU
