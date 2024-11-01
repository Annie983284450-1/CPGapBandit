#!/bin/bash
#SBATCH --job-name=xgbR1                          # Job name
#SBATCH -A gts-bzhao94                             # Charge account
#SBATCH -N1 -c12                       # Number of nodes and GPUs required
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32G
#SBATCH -t100:00:00                                       # Duration of the job  
#SBATCH -qinferno                                   # QOS name
#SBATCH --output=/storage/home/hcoda1/6/azhou60/scratch/cpbandit_results_shuffled/testSeptic250_trainSeptic1000_NoRefit_xgbcombo_ratio1.log  
#SBATCH --mail-type=BEGIN,END,FAIL                  # Mail preferences
#SBATCH --mail-user=azhou60@gatech.edu            # e-mail address for notifications


module load anaconda3
conda activate cpbanditpy39
cd /storage/home/hcoda1/6/azhou60/p-bzhao94-0/CPGapBandit_phoniex
srun python cpbandit_prob.py --num_test_sepsis_pat 250 --num_train_sepsis_pat 1000 --combo xgb 