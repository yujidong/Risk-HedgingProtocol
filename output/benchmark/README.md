# Benchmark Results Directory

This directory contains blockchain benchmark results from testnet deployments.

## File Naming Convention

```
benchmark_<network>_<timestamp>.json
```

- `network`: testnet identifier (e.g., "testnet" for Sepolia)
- `timestamp`: Unix timestamp in milliseconds

## Sample Results

Representative benchmark results are included in the repository for reference. Your actual results may vary based on:
- Network congestion
- Gas prices at time of execution
- Node performance

## Generating New Results

To generate your own benchmark results:

```bash
# Run blockchain benchmarks
npx hardhat test test/DataEquityProtocol.benchmark.js --network sepolia
```

Results will be automatically saved to this directory.

## Result Format

Each JSON file contains:
- Network information
- Test execution details
- Gas usage per operation
- Transaction costs (in ETH and USD)
- Performance metrics

Example structure:
```json
{
  "network": "sepolia",
  "timestamp": 1765792554132,
  "gasPrice": "...",
  "ethPrice": 3000,
  "results": {
    "deployment": { "gasUsed": 21000, "costETH": "...", "costUSD": "..." },
    "orderCreation": { "gasUsed": 273000, "costETH": "...", "costUSD": "..." },
    ...
  }
}
```

## Notes

⚠️ **File Size**: Benchmark files are typically < 50KB each
⚠️ **Git Ignore**: This directory is in `.gitignore` - results are regenerated on each run
✅ **Sample Preserved**: One representative sample is kept for documentation
