#!/bin/sh
#SBATCH --job-name=job2
#SBATCH --output=job3.out
#SBATCH --error=job3.err
#SBATCH --partition=cpu-fat
#SBATCH --ntasks=288
#SBATCH --time=48:00:00

module load gcc openmpi hdf5/serial python/3
source /home/elirans/project/queues/bin/activate
mpirun python3 //home/elirans/project/transient_inv/code/inventory_simpy_ph.py --n-settings 5000 --dynamic-demand --S-min 5 --S-max 15 --s-min 2 --s-max 14  --partition-simulations 5000 --replications 110000 --horizon 100 --inter-size 100 --lead-size 100 --control-max-tries 1000 --inv-dir /home/elirans/scratch/elad_trans/training_data/inv_level_3 --order-dir /home/elirans/scratch/elad_trans/training_data/order_3 --loss-dir /home/elirans/scratch/elad_trans/training_data/loss_3