#!/bin/bash -l
#SBATCH -p epyc -c 16 --mem 96gb --out logs/training_stage_annotation.log

CPU=${SLURM_CPUS_ON_NODE:-1}
echo $CPU
export PATH="$HOME/.pixi/bin:$PATH"
pixi run python train.py --mode annotation --data-dir classify \
	--model-dir models/annotation --n-workers $CPU

