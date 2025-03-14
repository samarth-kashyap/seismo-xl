#!/bin/bash
#SBATCH --account=seismo
#SBATCH --partition=swan
#SBATCH --qos=swan_default
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --job-name=n90s45
#SBATCH --output=logs/slurm_%x_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24

echo $SLURM_JOB_NAME
sleep 2
source /home/kashyap/.bashrc
conda activate py39cu121
/data/seismo/kashyap/opt/anaconda3/envs/py39cu121/bin/python -u /data/seismo/kashyap/codes/p11-seismo-xl/compute_fit_ps_8006161.py --Navg 90 --Nshift 45 --Nmcmc 20000 --peakbag &> /data/seismo/kashyap/codes/p11-seismo-xl/jobscripts/logs/n90s45.log
