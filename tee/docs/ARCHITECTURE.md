# Architecture Documentation

## System Overview

`
+-------------------------------------------------------+
|                Azure DC2s_v3 VM                       |
|              (Intel SGX Enabled)                      |
+-------------------------------------------------------+
|                                                       |
|  +-----------------+      +-----------------+         |
|  |  Native Docker  |      |  Gramine SGX    |         |
|  |  (Baseline)     |      |  (TEE Test)     |         |
|  +-----------------+      +-----------------+         |
|  | Python 3.10     |      | Python 3.12     |         |
|  | PyTorch 2.5.1   |      | PyTorch 2.5.1   |         |
|  | CPU-only        |      | CPU-only        |         |
|  |                 |      |                 |         |
|  | tee_benchmark.py|      | tee_benchmark.py|         |
|  +--------+--------+      +--------+--------+         |
|           |                        |                  |
|           |                   +----v-----+            |
|           |                   | Gramine  |            |
|           |                   | Runtime  |            |
|           |                   +----+-----+            |
|           |                        |                  |
|           |                   +----v-----+            |
|           |                   |SGX       |            |
|           |                   |Enclave   |            |
|           |                   |(8GB EPC) |            |
|           |                   +----------+            |
|           |                        |                  |
|           +------------+-----------+                  |
|                        |                              |
|                  +-----v------+                       |
|                  | Results    |                       |
|                  | (*.json)   |                       |
|                  +------------+                       |
+-------------------------------------------------------+
`

## Data Flow

### 1. Data Loading Phase

`
PEMS08.npz (17856 timesteps)
    |
    v
Load into NumPy array
    |
    v
Aggregate to 30-min intervals (1464 samples)
    |
    v
Create sliding windows (window_size=24)
    |
    v
Split train/test (80%/20%)
    |
    v
X: (1464, 24, 4590), Y: (1464, 170)
`

Native: 20.2s | SGX: 35.7s (+76%)

### 2. Model Inference Phase

`
Test Data (293 samples, batch_size=32)
    |
    v
LSTM Forward Pass
    |
    v
Predictions (293, 170)
    |
    v
Compute Metrics (R2, RMSE)
    |
    v
Web3 Signature (Ethereum signer)
`

Native: 547ms | SGX: 642ms (+17%)

### 3. Result Aggregation

`
10 Iterations x (Inference + Signature)
    |
    v
Statistics (Mean, Std, Min, Max)
    |
    v
Memory Tracking (psutil)
    |
    v
JSON Output
`

## SGX Enclave Memory Layout

`
+-------------------------------------+  8GB Total
|                                     |
|  +-------------------------------+  |
|  |  PyTorch Model + Weights      |  |  ~500MB
|  |  (traffic_lstm.pth)           |  |
|  +-------------------------------+  |
|                                     |
|  +-------------------------------+  |
|  |  Input Data (Batch)           |  |  ~200MB
|  |  (293 samples, 24x4590)       |  |
|  +-------------------------------+  |
|                                     |
|  +-------------------------------+  |
|  |  Intermediate Activations     |  |  ~800MB
|  |  (LSTM hidden states)         |  |
|  +-------------------------------+  |
|                                     |
|  +-------------------------------+  |
|  |  NumPy/Pandas Operations      |  |  ~900MB
|  |  (DataFrame, aggregation)     |  |
|  +-------------------------------+  |
|                                     |
|  +-------------------------------+  |
|  |  Python Runtime + Libraries   |  |  ~1GB
|  |  (interpreter, stdlib)        |  |
|  +-------------------------------+  |
|                                     |
|  Peak Usage: ~2.4GB (30% of 8GB)   |
|  Remaining: ~5.6GB buffer           |
+-------------------------------------+
`

**Why 8GB is necessary**:
- Native peak: 2.0GB
- SGX overhead: 1.2x-1.5x memory multiplier
- Safety margin: 2.4GB actual < 8GB allocated

**Why 2GB failed**:
- Enclave ran out of memory during pandas DataFrame creation
- NumPy array allocation exceeded available space

## Performance Bottlenecks

### Native Docker

| Component | Time | % of Total |
|-----------|------|------------|
| Data I/O (npz load) | 20.2s | 97.3% |
| Model inference | 0.55s | 2.6% |
| Signature | 0.01s | 0.1% |

### Gramine SGX

| Component | Time | Overhead | Reason |
|-----------|------|----------|---------|
| Data I/O | 35.7s | +76% | Encrypted memory, page faults |
| Model inference | 0.64s | +17% | Minimal CPU overhead |
| Signature | 0.01s | 0% | Pure computation |

**Key Insight**: SGX overhead is dominated by memory-intensive I/O, not computation.

## Critical Configuration Details

### 1. CPU-Only PyTorch

`dockerfile
# WRONG: Includes CUDA libraries (~900MB)
RUN pip3 install torch

# CORRECT: CPU-only (~175MB)
RUN pip3 install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
`

**Why**: SGX enclave cannot load CUDA shared libraries (libtorch_cuda.so).

### 2. Gramine Mode

`dockerfile
# WRONG: Simulation mode (no real SGX)
ENTRYPOINT ["gramine-direct", "tee_benchmark"]

# CORRECT: Real SGX hardware
ENTRYPOINT ["gramine-sgx", "tee_benchmark"]
`

### 3. SGX Device Mounts

`ash
# WRONG: Missing SGX devices
docker run tee-benchmark-gramine

# CORRECT: All required mounts
docker run --privileged \
  --device=/dev/sgx_enclave \
  --device=/dev/sgx_provision \
  -v /var/run/aesmd:/var/run/aesmd \
  tee-benchmark-gramine
`

### 4. Enclave Size

`	oml
# WRONG: Too small
sgx.enclave_size = "2G"  -> Memory error

# WRONG: Not power of 2
sgx.enclave_size = "6G"  -> "not a power of two"

# CORRECT: Power of 2, sufficient size
sgx.enclave_size = "8G"
`

## Gramine Manifest Sections

### Loader Configuration

`	oml
loader.argv = ["python3.12", "tee_benchmark.py", "--iterations", "10"]
loader.env.PYTHONPATH = "/app"
libos.entrypoint = "/usr/bin/python3.12"
fs.start_dir = "/app"
`

### File System Mounts

`	oml
fs.mounts = [
    { path = "/lib", uri = "file:{{ gramine.runtimedir() }}" },
    { path = "/app/data", uri = "file:/app/data" },
    { path = "/app/models", uri = "file:/app/models" },
    { path = "/app/results", type = "tmpfs" }  # <- Writable
]
`

### SGX Trusted Files

`	oml
sgx.trusted_files = [
    "file:/app/tee_benchmark.py",           # Application code
    "file:/usr/bin/python3.12",             # Python interpreter
    "file:/usr/lib/python3.12/",            # Python stdlib
    "file:/app/data/PEMS08.npz",            # Data (read-only)
    "file:/app/models/traffic_lstm.pth",    # Model (read-only)
]

sgx.allowed_files = [
    "file:/app/results/",  # Output directory (untrusted, writable)
]
`

**Security Note**: allowed_files passes data through without verification - use for outputs only.

## Docker Layer Optimization

`dockerfile
# Layer 1: Base image (cached)
FROM gramineproject/gramine:latest

# Layer 2: System packages (cached)
RUN apt-get update && apt-get install python3.10

# Layer 3: Python packages (cached, largest layer ~180MB)
RUN pip3 install numpy pandas torch==2.5.1 ...

# Layer 4-6: Application code (changes frequently, small)
COPY tee_benchmark.py .
COPY tee_benchmark.manifest.template .

# Layer 7-8: Data files (large, changes rarely)
COPY data/PEMS08.npz /app/data/
COPY models/traffic_lstm.pth /app/models/

# Layer 9-11: Gramine manifest generation (cached if data unchanged)
RUN gramine-manifest ...
RUN gramine-sgx-sign ...
`

Build Time: ~5 minutes first time, ~30 seconds for code-only changes.

## Testing Workflow

`
+--------------+
| Local Dev    |
| (PowerShell) |
+------+-------+
       |
       | redeploy_all.ps1
       | |
       | 1. Create tarball
       | 2. SCP to VM
       | 3. Extract
       | 4. Docker build
       v
+--------------+
|  Azure VM    |
| (Ubuntu 22)  |
+------+-------+
       |
       | run_cloud_test.sh
       | |
       | 1. Native test -> results/native_*.json
       | 2. SGX test -> results/gramine_*.json
       | 3. Analysis -> comparison table
       v
+--------------+
|   Results    |
|  (JSON files)|
+--------------+
       |
       | scp download
       v
+--------------+
|  Local Dev   |
| (Analysis)   |
+--------------+
`

## Error Handling

### Memory Errors

`
Error: Unable to allocate 116 MiB
Cause: sgx.enclave_size too small
Fix:   Increase to 8G or higher
`

### CUDA Errors

`
Error: libtorch_cuda.so: failed to map segment
Cause: CUDA PyTorch in SGX enclave
Fix:   Install CPU-only: torch==2.5.1 --index-url .../cpu
`

### File Not Found

`
Error: Data file not found at data/PEMS08.npz
Cause: Docker working directory is /, not /app
Fix:   Add 'cd /app &&' before python command
`

### Simulation Mode

`
Symptom: No performance overhead (~0%)
Cause:   Using gramine-direct (simulation)
Fix:     Use gramine-sgx in ENTRYPOINT
`

## Performance Tuning

### Reduce Memory Usage

1. Reduce batch size: --batch-size 16 (default: 32)
2. Use low-memory mode: --low-memory (chunks data processing)
3. Reduce iterations: --iterations 5 (default: 10)

### Speed Up Tests

1. Reduce data size: --data-size 50 (use 50% of dataset)
2. Skip signature: Comment out web3 signing code
3. Use smaller model: Reduce hidden_size from 256 to 128

### Production Optimizations

1. Disable debug: Set sgx.debug = false in manifest
2. Use production keys: Generate real SGX signing keys
3. Enable AVX: Add sgx.cpu_features.avx = "required"

## Monitoring and Debugging

### Check SGX Status on VM

`ash
# Check SGX driver
ls -l /dev/sgx*

# Check AESM service
systemctl status aesmd

# Check SGX capabilities
cpuid | grep -i sgx
`

### Monitor Resources During Test

`ash
# In separate SSH session
watch -n 1 'docker stats --no-stream'
`

### Enable Verbose Logging

`	oml
# In manifest template
loader.log_level = "debug"  # Shows all Gramine operations
`

---

**Last Updated**: December 15, 2025
