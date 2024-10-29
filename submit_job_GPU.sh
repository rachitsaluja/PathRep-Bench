#!/bin/bash

mkdir -p job_err_EXP
mkdir -p job_out_EXP

sbatch --requeue \
        -p sablab-gpu \
        -t 168:00:00 \
        -n 20 \
        -N 1 \
        --nodelist=ai-gpu07 \
        --mem=50G \
        --gres=gpu:a100:1 \
        --mail-type=ALL \
        --mail-user=ras4037@med.cornell.edu \
        --job-name=$1 \
        -e ./job_err_EXP/%j-$1.err \
        -o ./job_out_EXP/%j-$1.out \
        $2