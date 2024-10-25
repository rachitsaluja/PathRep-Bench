#!/bin/bash

mkdir -p job_err_EXP
mkdir -p job_out_EXP

sbatch --requeue \
    -p sablab-highprio \
    -t 168:00:00 \
    -n 4 \
    -N 1 \
    --nodelist=sablab-cpu-01 \
    --mem=16G \
    --mail-type=ALL \
    --mail-user=rs2492@cornell.edu \
    --job-name=$1 \
    -e ./job_err_EXP/%j-$1.err \
    -o ./job_out_EXP/%j-$1.out \
    $2