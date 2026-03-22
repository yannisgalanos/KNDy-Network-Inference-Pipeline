#!/bin/bash
#$ -N kndy_calibration
#$ -o logs/calibration_$JOB_ID.out
#$ -e logs/calibration_$JOB_ID.err
#$ -j n
#$ -pe sharedmem 16
#$ -l h_rt=04:00:00
#$ -l h_vmem=8G
#$ -cwd

# ── Email notifications (optional — edit your address) ───────────────
#$ -M s2420052@ed.ac.uk
#$ -m bea

# ── Environment ──────────────────────────────────────────────────────
source /exports/eddie/scratch/$USER/kndy_env/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p logs calibration

echo "======================================"
echo "  KNDy Calibration Check"
echo "  Job ID:    $JOB_ID"
echo "  Host:      $(hostname)"
echo "  Cores:     $NSLOTS"
echo "  Started:   $(date)"
echo "======================================"

python calibration_check.py \
    --n-samples 500 \
    --n-workers $NSLOTS \
    --calcium-params calcium_params.json \
    --output calibration/ \
    --seed 42

echo ""
echo "======================================"
echo "  Calibration complete: $(date)"
echo "  Output: calibration/"
echo "======================================"
