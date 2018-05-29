#!/bin/bash
#
#SBATCH --partition=titanx-long    # Partition to submit to
#SBATCH --time=6-01:00:00
#SBATCH --gres=gpu:1
#

python $1 -s $2 -p $3
