#!/bin/sh
#SBATCH --job-name=job2
#SBATCH --output=job4.out
#SBATCH --error=job4.err
#SBATCH --partition=cpu
#SBATCH --ntasks=180
#SBATCH --time=48:00:00

module load gcc openmpi hdf5/serial python/3
source /home/elirans/project/queues/bin/activate
mpirun python3 //home/elirans/project/transient_inv/code/inventory_simpy_ph.py --n-settings 5000 --dynamic-demand --S-min 5 --S-max 30 --s-min 2 --s-max 29  --partition-simulations 5000 --replications 110000 --horizon 100 --inter-size 100 --lead-size 100 --control-max-tries 1000 --inv-dir /home/elirans/scratch/elad_trans/training_data/inv_level_7 --order-dir /home/elirans/scratch/elad_trans/training_data/order_7 --loss-dir /home/elirans/scratch/elad_trans/training_data/loss_7