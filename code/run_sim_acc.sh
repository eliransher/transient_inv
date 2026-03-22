#!/bin/bash
#SBATCH -t 0-03:58
#SBATCH -A def-dkrass
#SBATCH --mem 20000
source /home/eliransc/projects/def-dkrass/eliransc/mom_match/bin/activate
python /home/eliransc/projects/def-dkrass/eliransc/transient_inv/code/inventory_accuracy_trials.py --trials 10 --replications 500 --inv-dump-dir /scratch/eliransc/elad_trans/inv_level --order-dump-dir /scratch/eliransc/elad_trans/order --loss-dump-dir /scratch/eliransc/elad_trans/loss