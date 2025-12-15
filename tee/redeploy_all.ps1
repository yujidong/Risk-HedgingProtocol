# Complete Redeployment Script - Fixes all issues and redeploys to Azure
# Run this script to package and upload fixed files to Azure VM

$ErrorActionPreference = "Stop"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "COMPLETE REDEPLOYMENT - All Fixes Applied" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$packageName = "tee-benchmark-fixed-${timestamp}.tar.gz"
$vmIP = "172.190.61.89"
$vmUser = "azureuser"

# Step 1: Validate files exist
Write-Host "[1/5] Validating files..." -ForegroundColor Yellow
$requiredFiles = @(
    "Dockerfile.gramine",
    "tee_benchmark.py",
    "tee_benchmark.manifest.template",
    "run_cloud_test.sh",
    "requirements.txt",
    "data/PEMS08.npz",
    "models/traffic_lstm.pth"
)

foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        Write-Host "ERROR: Missing file: $file" -ForegroundColor Red
        exit 1
    }
}
Write-Host "  All required files present" -ForegroundColor Green

# Step 2: Create package
Write-Host "`n[2/5] Creating deployment package..." -ForegroundColor Yellow
tar -czf $packageName `
    Dockerfile.gramine `
    tee_benchmark.py `
    tee_benchmark.manifest.template `
    run_cloud_test.sh `
    requirements.txt `
    data/PEMS08.npz `
    models/traffic_lstm.pth

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to create package" -ForegroundColor Red
    exit 1
}

$packageSize = (Get-Item $packageName).Length / 1MB
Write-Host "  Package created: $packageName ($([math]::Round($packageSize, 2)) MB)" -ForegroundColor Green

# Step 3: Upload to Azure VM
Write-Host "`n[3/5] Uploading to Azure VM..." -ForegroundColor Yellow
scp $packageName "${vmUser}@${vmIP}:~/"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to upload" -ForegroundColor Red
    exit 1
}
Write-Host "  Upload complete" -ForegroundColor Green

# Step 4: Extract and rebuild on VM
Write-Host "`n[4/5] Extracting and rebuilding on VM..." -ForegroundColor Yellow

# Use separate SSH commands to avoid line ending issues
ssh "${vmUser}@${vmIP}" "cd ~ && rm -rf data models Dockerfile.gramine tee_benchmark.* run_cloud_test.sh requirements.txt"
ssh "${vmUser}@${vmIP}" "cd ~ && tar -xzf $packageName"
ssh "${vmUser}@${vmIP}" "cd ~ && docker build -f Dockerfile.gramine -t tee-benchmark-gramine:latest ."

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Remote build failed" -ForegroundColor Red
    exit 1
}
Write-Host "  Docker image rebuilt successfully" -ForegroundColor Green

# Step 5: Instructions
Write-Host "`n[5/5] Deployment Complete!" -ForegroundColor Green
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "NEXT STEPS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`n1. Connect to VM:" -ForegroundColor Yellow
Write-Host "   ssh ${vmUser}@${vmIP}" -ForegroundColor White
Write-Host "`n2. Run complete test:" -ForegroundColor Yellow
Write-Host "   ./run_cloud_test.sh" -ForegroundColor White
Write-Host "`n3. Or run tests individually:" -ForegroundColor Yellow
Write-Host "   # Native test" -ForegroundColor Gray
Write-Host "   docker run --rm -v `"`$(pwd)/results:/app/results`" -v `"`$(pwd)/data:/app/data:ro`" -v `"`$(pwd)/models:/app/models:ro`" -v `"`$(pwd)/tee_benchmark.py:/app/tee_benchmark.py:ro`" -v `"`$(pwd)/requirements.txt:/app/requirements.txt:ro`" python:3.10-slim bash -c `"pip install --quiet -r /app/requirements.txt && cd /app && python tee_benchmark.py --iterations 10 --output results/native.json`"" -ForegroundColor Gray
Write-Host "`n   # Gramine SGX test" -ForegroundColor Gray
Write-Host "   docker run --rm --privileged --device=/dev/sgx_enclave --device=/dev/sgx_provision -v /var/run/aesmd:/var/run/aesmd -v `"`$(pwd)/results:/app/results`" tee-benchmark-gramine:latest" -ForegroundColor Gray
Write-Host "`n4. Download results:" -ForegroundColor Yellow
Write-Host "   scp ${vmUser}@${vmIP}:~/results/*.json ." -ForegroundColor White
Write-Host "`n5. Stop VM when done:" -ForegroundColor Yellow
Write-Host "   az vm deallocate --resource-group web3 --name tee-benchmark-vm" -ForegroundColor White
Write-Host "`n========================================`n" -ForegroundColor Cyan

Write-Host "Estimated test time: 10-15 minutes" -ForegroundColor Yellow
Write-Host "Cost while running: `$0.248/hour (~`$0.04 for complete test)`n" -ForegroundColor Yellow
