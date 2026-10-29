#!/bin/bash
#SBATCH --partition=<FILL>
#SBATCH --gres=gpu:<FILL>:1
#SBATCH --time=<FILL>            # e.g. 23:59:00; the job resubmits itself, the pipeline resumes
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=covert
#SBATCH --output=runs/slurm-%j.out
source ~/.bashrc
conda activate <FILL>
cd "$(dirname "$0")/../.."
export HF_TOKEN WANDB_API_KEY            # set in your shell / ~/.bashrc, never in this file
# queue a follow-up job that starts after this one ends; if everything is finished it exits immediately (stages skip)
sbatch --dependency=afterany:$SLURM_JOB_ID "$0" >/dev/null 2>&1 || true
covert pipeline run
