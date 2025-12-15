# Run benchmarks on all testnets
# Usage: .\scripts\run_all_benchmarks.ps1

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Running Benchmarks on All Testnets" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Sepolia
Write-Host "📊 Testing on Sepolia..." -ForegroundColor Yellow
npx hardhat test test/DataEquityProtocol.benchmark.js --network sepolia
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Sepolia benchmark completed" -ForegroundColor Green
} else {
    Write-Host "❌ Sepolia benchmark failed" -ForegroundColor Red
}
Write-Host ""

# Arbitrum Sepolia
Write-Host "📊 Testing on Arbitrum Sepolia..." -ForegroundColor Yellow
npx hardhat test test/DataEquityProtocol.benchmark.js --network arbitrumSepolia
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Arbitrum Sepolia benchmark completed" -ForegroundColor Green
} else {
    Write-Host "❌ Arbitrum Sepolia benchmark failed" -ForegroundColor Red
}
Write-Host ""

# Optimism Sepolia
Write-Host "📊 Testing on Optimism Sepolia..." -ForegroundColor Yellow
npx hardhat test test/DataEquityProtocol.benchmark.js --network optimismSepolia
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Optimism Sepolia benchmark completed" -ForegroundColor Green
} else {
    Write-Host "❌ Optimism Sepolia benchmark failed" -ForegroundColor Red
}
Write-Host ""

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "All benchmarks completed!" -ForegroundColor Cyan
Write-Host "Results saved to: output/benchmark/" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
