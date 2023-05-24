#!/bin/bash

#SBATCH --job-name=dfe_sim_1B08_2e6_contig
#SBATCH --output=outfiles/%x-%j.out
#SBATCH --error=outfiles/%x-%j.err
#SBATCH --account=rgutenk
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_rgutenk
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lnt@arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=90
#SBATCH --mem=450gb
#SBATCH --time=48:00:00

date

source ~/.bashrc && conda activate dfe-cnn
python simulation.py

date
