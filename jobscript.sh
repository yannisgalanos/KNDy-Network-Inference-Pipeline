#!/bin/bash
#
# SGE job script for KNDy SNPE training + inference
# Submit with:  qsub jobscript.sh
#

# ── Job name and output ──────────────────────────────────────────────
#$ -N kndy_snpe
#$ -o logs/kndy_snpe_$JOB_ID.out
#$ -e logs/kndy_snpe_$JOB_ID.err
#$ -j n

# ── Resources ────────────────────────────────────────────────────────
# Request a single node with 4 CPU cores.
# Each simulation is sequential, but sklearn and numpy use threads.
#$ -pe sharedmem 32

# Wall time: ~15 hours covers 1500 training sims + inference + PPC.
# Adjust based on your calibration timing (~1 min per simulation).
#$ -l h_rt=04:00:00

# Memory: 8GB should be plenty for 100 neurons.
#$ -l h_vmem=12G

# ── Email notifications (optional — edit your address) ───────────────
#$ -M s2420052@ed.ac.uk
#$ -m bea

# ── Queue (uncomment and edit if your cluster has named queues) ──────
# #$ -q long.q

# ── Working directory ────────────────────────────────────────────────
#$ -cwd

# ── Environment setup ────────────────────────────────────────────────
# Uncomment and edit whichever applies to your cluster:

# Option A: module system
# module load python/3.10
# module load gcc       # needed for the C++ extension (mycpp)

# Option B: conda
# source activate kndy_env

# Option C: venv
source /exports/eddie/scratch/$USER/kndy_env/bin/activate

# When using multiprocessing, each worker should use 1 thread
# to avoid oversubscription (4 workers x 4 threads = 16 on 4 cores)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ── Create output directories ────────────────────────────────────────
mkdir -p logs models results

# ── Print job info ───────────────────────────────────────────────────
echo "======================================"
echo "  KNDy SNPE Job"
echo "  Job ID:    $JOB_ID"
echo "  Host:      $(hostname)"
echo "  Cores:     $NSLOTS"
echo "  Started:   $(date)"
echo "======================================"
echo ""

# ── Stage 0: Build C++ extension ─────────────────────────────────────
# Only needed once — if mycpp is already compiled, this is fast.
echo "[$(date +%H:%M:%S)] Building C++ extension ..."
bash build_mycpp.sh
python3 -c "import mycpp; print('  mycpp imported OK')"

BUILD_EXIT=$?
if [ $BUILD_EXIT -ne 0 ]; then
    echo "ERROR: C++ build or import failed with exit code $BUILD_EXIT"
    exit $BUILD_EXIT
fi
echo ""

# ── Stage 1: Train the amortised model ───────────────────────────────
# This is the expensive step (~10-15 hours for 1500 simulations).
# The model is saved to models/ and can be reused for any data.
echo "[$(date +%H:%M:%S)] Starting training ..."

python train_model.py \
    --num-simulations 1500 \
    --n-workers $NSLOTS \
    --output models/ \
    --seed 42

TRAIN_EXIT=$?
if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: Training failed with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

echo ""
echo "[$(date +%H:%M:%S)] Training complete."
echo ""

# ── Stage 2: Infer parameters from data ──────────────────────────────
# This uses the trained model — conditioning is instant,
# only the PPC simulations take time (~20 min for 20 sims).
#
# For real data, replace --demo with:
#   --spikes /path/to/your/spikes.npy
echo "[$(date +%H:%M:%S)] Starting inference ..."

python infer.py \
    --model-dir models/ \
    --demo \
    --output results/ \
    --n-samples 1000 \
    --n-ppc 30

INFER_EXIT=$?
if [ $INFER_EXIT -ne 0 ]; then
    echo "ERROR: Inference failed with exit code $INFER_EXIT"
    exit $INFER_EXIT
fi

echo ""
echo "======================================"
echo "  Job complete: $(date)"
echo "  Output: results/"
echo "  Model:  models/"
echo "======================================"
