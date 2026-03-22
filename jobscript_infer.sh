#!/bin/bash
#
# SGE job script for KNDy inference on real data
# Submit with:  qsub jobscript_infer.sh
#
# Edit the --traces path below before submitting.
#

#$ -N kndy_infer
#$ -o logs/infer_$JOB_ID.out
#$ -e logs/infer_$JOB_ID.err
#$ -j n
#$ -pe sharedmem 4
#$ -l h_rt=01:00:00
#$ -l h_vmem=4G
#$ -cwd

source /exports/eddie/scratch/$USER/kndy_env/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p logs results_real

echo "======================================"
echo "  KNDy Inference on Real Data"
echo "  Job ID:    $JOB_ID"
echo "  Started:   $(date)"
echo "======================================"

python infer.py \
    --model-dir models/ \
    --traces data/G14D_2024_02_09_raw.csv \
    --output results_real/ \
    --n-samples 1000 \
    --n-ppc 30

echo ""
echo "======================================"
echo "  Inference complete: $(date)"
echo "  Output: results_real/"
echo "======================================"
