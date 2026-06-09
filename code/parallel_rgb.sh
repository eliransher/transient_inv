#!/bin/sh
#SBATCH --job-name=job2
#SBATCH --output=job2.out
#SBATCH --error=job2.err
#SBATCH --partition=cpu_fat
#SBATCH --ntasks=64
#SBATCH --time=48:00:00

module load gcc openmpi hdf5/serial python/3
source /home/elirans/project/queues/bin/activate
mpirun python3 //home/elirans/project/transient_inv/code/inventory_simpy_ph.py --dynamic-demand --n-settings 454 --replications 110000 --horizon 100 --inv-dir /home/elirans/scratch/elad_trans/training_data/inv_level --order-dir /home/elirans/scratch/elad_trans/training_data/order --loss-dir /home/elirans/scratch/elad_trans/training_data/loss