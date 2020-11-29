#!/bin/bash
#SBATCH -A INF20_lqcd123_0
# ##SBATCH -p m100_all_serial
#SBATCH -p m100_usr_prod
#SBATCH --time 04:00:00     # format: HH:MM:SS
#SBATCH -N 1  --exclusive   # 1 node
#SBATCH --ntasks-per-node=1 # 8 tasks out of 128
#SBATCH --gres=gpu:1        # 1 gpus per node out of 4
#SBATCH --mem=246000          # memory per node out of 246000MB
#SBATCH --job-name=benchmark
#  #SBATCH --mail-type=ALL
#  #SBATCH --mail-user=<user_email>

#if [ -z "${2}" ]; then
#  echo usage: sbatch ${0} executable  outfile
#  exit 1
#fi


source  ../load_modules.sh
#module list > $2

export OMP_PROC_BIND=true 
export OMP_PLACES=cores

srun  ./main -i example.in &> output_GPU
#  &> redirect both stdout and stderr in to a file

#interactive node
#salloc -N1 --exclusive --gres=gpu:1 -A INF20_lqcd123_0  -p m100_usr_prod --time=00:10:00
#export OMP_PROC_BIND=true  OMP_PLACES=cores
#OMP_NUM_THREADS=6
