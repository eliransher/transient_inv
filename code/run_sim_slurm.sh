#!/bin/sh
#SBATCH --partition=cpu
#SBATCH --job-name=job1
#SBATCH --output=job1.out
#SBATCH --error=job1.err
#SBATCH --ntasks=1
#SBATCH --time=01:00:00


source /home/elirans/project/queues/bin/activate
python //home/elirans/project/transient_inv/code/inventory_simpy_ph.py --dynamic-demand --n-settings 454 --replications 110000 --horizon 100 --inv-dir /home/elirans/scratch/elad_trans/training_data/inv_level --order-dir /home/elirans/scratch/elad_trans/training_data/order --loss-dir /home/elirans/scratch/elad_trans/training_data/loss