#!/bin/bash -l
#SBATCH -p exfab --gres=gpu:1 -c 8 --mem 96gb --out logs/training_stage.log

CPU=${SLURM_CPUS_ON_NODE:-1}
module load cuda/12.8
module load cudnn
export PATH="$HOME/.pixi/bin:$PATH"
pixi run python train.py --mode embedding --seq-type cds --data-dir classify --model-dir models/evo2_cds \
	--embedding-cache models/embeddings --n-workers $CPU


