# TEE Benchmark Results Directory

This directory contains performance benchmark results from Trusted Execution Environment (TEE) testing.

## Execution Modes

### 1. Native Mode (`native_*.json`)
- Direct Python execution without TEE
- Baseline performance metrics
- Used for comparison

### 2. Gramine-SGX Mode (`gramine_*.json`)
- Intel SGX with Gramine framework
- Security-enhanced execution
- Production-ready configuration

## Result Types

### Manual Benchmarks (`*_manual_*.json`)
- Single iteration measurements
- Detailed step-by-step timing
- Used for debugging and analysis

### Automated Benchmarks (`*_auto_*.json`)
- Multiple iterations (10+)
- Statistical aggregation (mean, std, min, max)
- Production-grade measurements

## File Naming

```
<mode>_<type>_<timestamp>.json
```

- `mode`: native | gramine_sync | gramine_auto
- `type`: manual | auto | (iteration count)
- `timestamp`: YYYYMMDD_HHMMSS

## Key Metrics

Each benchmark file contains:

```json
{
  "execution_mode": "native" | "gramine",
  "timestamp": "...",
  "system_info": { "cpu": "...", "memory": "...", "sgx_available": true/false },
  "performance": {
    "total_time": 123.45,
    "overhead": 0.17,  // TEE overhead vs native
    "steps": {
      "data_loading": 1.23,
      "model_inference": 45.67,
      "signing": 0.89
    }
  },
  "accuracy": {
    "r2_score": 0.95,
    "mse": 0.001
  }
}
```

## Expected Results

| Metric | Native | Gramine-SGX | Overhead |
|--------|--------|-------------|----------|
| Total Time | ~100s | ~117s | ~17% |
| Accuracy | R²=0.95 | R²=0.95 | 0% |
| Memory | 2GB | 2.5GB | +25% |

## Running Benchmarks

### Local (Native)
```bash
cd tee/
python tee_benchmark.py
```

### Azure SGX VM
```bash
cd tee/
.\create_azure_vm.ps1  # Create VM
.\redeploy_all.ps1     # Deploy and run
```

## Cloud Testing

Results from Azure DC-series VMs:
- **VM Size**: Standard_DC2s_v3
- **SGX EPC**: 8GB encrypted memory
- **CPU**: 2 vCPUs (Intel Xeon with SGX)
- **Location**: East US, West Europe, UK South

## Notes

⚠️ **Hardware Dependent**: SGX results only valid on supported hardware
⚠️ **Overhead Variability**: TEE overhead varies by:
  - Workload characteristics
  - Data size
  - Memory access patterns
  - SGX EPC size

✅ **Reproducibility**: Statistical benchmarks (10+ iterations) provide reliable estimates
✅ **Security**: TEE guarantees identical accuracy with confidentiality
