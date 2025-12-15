#!/bin/bash
# ============================================================================
# Automated TEE Performance Test - For Linux/WSL/Ubuntu Cloud Server
# Usage: ./run_cloud_test.sh
# ============================================================================

set -e

echo "============================================================"
echo "TEE Performance Comparison Test"
echo "============================================================"
echo ""

timestamp=$(date +%Y%m%d_%H%M%S)
native_output="native_auto_${timestamp}.json"
gramine_output="gramine_auto_${timestamp}.json"

echo "Timestamp: ${timestamp}"
echo ""

mkdir -p results

# Step 1: Native test
echo "[1/3] Running Native test..."
docker run --rm \
    -v "$(pwd)/results:/app/results" \
    -v "$(pwd)/data:/app/data:ro" \
    -v "$(pwd)/models:/app/models:ro" \
    -v "$(pwd)/tee_benchmark.py:/app/tee_benchmark.py:ro" \
    -v "$(pwd)/requirements.txt:/app/requirements.txt:ro" \
    python:3.10-slim \
    bash -c "
        pip install --quiet -r /app/requirements.txt && \
        cd /app && \
        python tee_benchmark.py \
            --iterations 10 \
            --data-size 100 \
            --batch-size 32 \
            --output results/${native_output}
    "

[ -f "results/${native_output}" ] || { echo "ERROR: Native test failed"; exit 1; }
echo "Native completed"
echo ""

# Step 2: Gramine SGX test
echo "[2/3] Running Gramine SGX test..."
docker run --rm \
    --privileged \
    --device=/dev/sgx_enclave \
    --device=/dev/sgx_provision \
    -v /var/run/aesmd:/var/run/aesmd \
    -v "$(pwd)/results:/app/results" \
    tee-benchmark-gramine:latest

# Rename output to match timestamp
mv results/tee_benchmark_results.json "results/${gramine_output}" 2>/dev/null || true

[ -f "results/${gramine_output}" ] || { echo "ERROR: Gramine test failed"; exit 1; }
echo "Gramine completed"
echo ""

# Step 3: Analyze
echo "[3/3] Analyzing results..."
docker run --rm \
    -v "$(pwd)/results:/results:ro" \
    python:3.10-slim \
    bash -c "
pip install --quiet numpy && python3 -c \"
import json
import numpy as np

with open('/results/${native_output}') as f:
    nd = json.load(f)
with open('/results/${gramine_output}') as f:
    gd = json.load(f)

# Extract timings (in seconds, convert to ms)
nt_list = nd['timings']['inference']
gt_list = gd['timings']['inference']

nt = np.mean(nt_list) * 1000
ns = np.std(nt_list) * 1000
gt = np.mean(gt_list) * 1000
gs = np.std(gt_list) * 1000

nr2 = nd['metrics']['r2_score'][0]
gr2 = gd['metrics']['r2_score'][0]

nmem = nd['memory']['peak_mb'] - nd['memory']['initial_mb']
gmem = gd['memory']['peak_mb'] - gd['memory']['initial_mb']

oh_time = gt - nt
oh_pct = (oh_time / nt) * 100
oh_mem = gmem - nmem

print('=' * 70)
print('TEE PERFORMANCE COMPARISON RESULTS')
print('=' * 70)
print()
print(f'Inference Time:')
print(f'  Native:   {nt:.2f} +/- {ns:.2f} ms')
print(f'  Gramine:  {gt:.2f} +/- {gs:.2f} ms')
print(f'  Overhead: +{oh_time:.2f} ms (+{oh_pct:.1f}%)')
print()
print(f'Model Accuracy (R2 Score):')
print(f'  Native:   {nr2:.4f}')
print(f'  Gramine:  {gr2:.4f}')
print(f'  Diff:     {abs(gr2-nr2):.6f} (identical: {nr2==gr2})')
print()
print(f'Memory Usage:')
print(f'  Native:   {nmem:.0f} MB')
print(f'  Gramine:  {gmem:.0f} MB')
print(f'  Overhead: +{oh_mem:.0f} MB')
print()
print('=' * 70)
print(f'Native:  {nd[\\\"config\\\"][\\\"timestamp\\\"]}')
print(f'Gramine: {gd[\\\"config\\\"][\\\"timestamp\\\"]}')
print('=' * 70)
\"
"

echo ""
echo "Done! Results saved to results/"
