#!/bin/bash
#SBATCH -t 0-23:58
#SBATCH -A def-dkrass
#SBATCH --mem 20000
source /home/eliransc/projects/def-dkrass/eliransc/mom_match/bin/activate
python /home/eliransc/projects/def-dkrass/eliransc/transient_inv/code/inventory_simpy_ph.py --dynamic-demand --n-settings 3000 --replications 75000 --horizon 100 --inv-dir /scratch/eliransc/elad_trans/train_data/inv_level --order-dir /scratch/eliransc/elad_trans/train_data/order --loss-dir /scratch/eliransc/elad_trans/train_data/loss