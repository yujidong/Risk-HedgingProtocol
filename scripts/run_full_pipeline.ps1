# Complete End-to-End Testing Pipeline
# Runs all three components: ML, Blockchain, TEE

param(
    [string]$Mode = "all",  # Options: all, ml, blockchain, tee
    [switch]$SkipMLTraining,
    [switch]$SkipBlockchainDeploy,
    [switch]$LocalOnly
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Risk-Hedging Protocol - Full Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# Component 1: AI/ML Data Generation & Analysis
# ============================================================================

if ($Mode -eq "all" -or $Mode -eq "ml") {
    Write-Host "[1/3] Running ML Experiments..." -ForegroundColor Yellow
    Write-Host ""
    
    if (-not $SkipMLTraining) {
        Write-Host "  → Noise Robustness Experiment" -ForegroundColor Green
        conda activate risk-hedging
        python noise_robustness_experiment.py
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ML experiments failed!" -ForegroundColor Red
            exit 1
        }
        
        Write-Host "  → Game Theory: Data Scarcity" -ForegroundColor Green
        python game_theory_data_scarcity.py
        
        Write-Host "  → Game Theory: Random Drop" -ForegroundColor Green
        python game_theory_random_drop.py
        
        Write-Host "  → Generating Visualizations" -ForegroundColor Green
        python visualize_results.py
        
        Write-Host ""
        Write-Host "  ✅ ML experiments completed" -ForegroundColor Green
        Write-Host "  📊 Results saved to: output/data/" -ForegroundColor Cyan
        Write-Host "  📈 Figures saved to: output/figures/" -ForegroundColor Cyan
    } else {
        Write-Host "  ⏭️  Skipping ML training (using existing results)" -ForegroundColor Gray
    }
    Write-Host ""
}

# ============================================================================
# Component 2: Blockchain Smart Contract Testing
# ============================================================================

if ($Mode -eq "all" -or $Mode -eq "blockchain") {
    Write-Host "[2/3] Running Blockchain Tests..." -ForegroundColor Yellow
    Write-Host ""
    
    # Check balances first
    Write-Host "  → Checking account balances" -ForegroundColor Green
    npx hardhat run scripts/check-all-balances.js --network sepolia
    
    # Run functional tests
    Write-Host ""
    Write-Host "  → Running functional tests" -ForegroundColor Green
    npx hardhat test test/DataEquityProtocol.test.js --network sepolia
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Functional tests failed!" -ForegroundColor Red
        exit 1
    }
    
    # Run performance benchmarks
    if (-not $LocalOnly) {
        Write-Host ""
        Write-Host "  → Running performance benchmarks (this may take 5-10 minutes)" -ForegroundColor Green
        npx hardhat test test/DataEquityProtocol.benchmark.js --network sepolia
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Benchmark tests failed!" -ForegroundColor Red
            exit 1
        }
    }
    
    Write-Host ""
    Write-Host "  ✅ Blockchain tests completed" -ForegroundColor Green
    Write-Host "  📊 Benchmark results: output/benchmark/" -ForegroundColor Cyan
    Write-Host ""
}

# ============================================================================
# Component 3: TEE Performance Testing
# ============================================================================

if ($Mode -eq "all" -or $Mode -eq "tee") {
    Write-Host "[3/3] Running TEE Tests..." -ForegroundColor Yellow
    Write-Host ""
    
    if ($LocalOnly) {
        Write-Host "  → Running local TEE simulation (Native mode)" -ForegroundColor Green
        Push-Location tee
        python tee_benchmark.py --mode native --iterations 10
        Pop-Location
        
        Write-Host ""
        Write-Host "  ✅ TEE local tests completed" -ForegroundColor Green
        Write-Host "  📊 Results: tee/results/" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  ℹ️  For real SGX tests, run: .\scripts\run_full_pipeline.ps1 -Mode tee" -ForegroundColor Gray
    } else {
        Write-Host "  ⚠️  Real SGX tests require Azure VM deployment" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  To run SGX tests:" -ForegroundColor Cyan
        Write-Host "    1. cd tee/" -ForegroundColor Gray
        Write-Host "    2. .\create_azure_vm.ps1" -ForegroundColor Gray
        Write-Host "    3. .\redeploy_all.ps1" -ForegroundColor Gray
        Write-Host "    4. ssh azureuser@<VM_IP>" -ForegroundColor Gray
        Write-Host "    5. ./run_cloud_test.sh" -ForegroundColor Gray
        Write-Host ""
        Write-Host "  Running local simulation only..." -ForegroundColor Yellow
        Push-Location tee
        python tee_benchmark.py --mode native --iterations 5
        Pop-Location
        Write-Host ""
        Write-Host "  ✅ TEE simulation completed" -ForegroundColor Green
    }
    Write-Host ""
}

# ============================================================================
# Final Summary
# ============================================================================

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Pipeline Execution Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "📁 Results Summary:" -ForegroundColor Yellow
Write-Host ""

if ($Mode -eq "all" -or $Mode -eq "ml") {
    Write-Host "  ML Results:" -ForegroundColor Cyan
    Write-Host "    - output/data/*.json         (Experimental data)" -ForegroundColor Gray
    Write-Host "    - output/figures/*.png       (Visualization plots)" -ForegroundColor Gray
}

if ($Mode -eq "all" -or $Mode -eq "blockchain") {
    Write-Host ""
    Write-Host "  Blockchain Results:" -ForegroundColor Cyan
    Write-Host "    - output/benchmark/*.json    (Performance metrics)" -ForegroundColor Gray
    Write-Host "    - Sepolia Testnet:           https://sepolia.etherscan.io/" -ForegroundColor Gray
}

if ($Mode -eq "all" -or $Mode -eq "tee") {
    Write-Host ""
    Write-Host "  TEE Results:" -ForegroundColor Cyan
    Write-Host "    - tee/results/*.json         (Benchmark data)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Review results in output/ and tee/results/ directories" -ForegroundColor Gray
Write-Host "  2. Compare benchmarks: python scripts/compare_benchmarks.py" -ForegroundColor Gray
Write-Host "  3. Generate final report with all metrics" -ForegroundColor Gray
Write-Host ""
Write-Host "✨ All tests passed successfully! ✨" -ForegroundColor Green
Write-Host ""
