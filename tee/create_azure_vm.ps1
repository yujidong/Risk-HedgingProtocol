#!/usr/bin/env pwsh
# ============================================================================
# Azure VM Creation Script for TEE Benchmark Testing
# Resource Group: web3
# VM Size: Standard_DC2s_v3 (2 vCPU, 16GB RAM, 8GB SGX EPC)
# ============================================================================

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Creating Azure VM for TEE Benchmark Testing" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$resourceGroup = "web3"
$vmName = "tee-benchmark-vm"
$vmSize = "Standard_DC2s_v3"
$location = "eastus"  # DC-series available in: eastus, westeurope, uksouth
$adminUsername = "azureuser"
$ubuntuImage = "Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest"

# Check if logged in
Write-Host "[1/6] Checking Azure login status..." -ForegroundColor Yellow
try {
    $account = az account show 2>&1 | ConvertFrom-Json
    Write-Host "  [OK] Logged in as: $($account.user.name)" -ForegroundColor Green
    Write-Host "  Subscription: $($account.name)" -ForegroundColor Gray
} catch {
    Write-Host "  [ERROR] Not logged in to Azure" -ForegroundColor Red
    Write-Host "  Please run: az login" -ForegroundColor Yellow
    exit 1
}

# Check if resource group exists
Write-Host "`n[2/6] Checking resource group..." -ForegroundColor Yellow
$rgExists = az group exists --name $resourceGroup
if ($rgExists -eq "true") {
    Write-Host "  [OK] Resource group '$resourceGroup' exists" -ForegroundColor Green
} else {
    Write-Host "  [WARNING] Resource group '$resourceGroup' does not exist" -ForegroundColor Red
    Write-Host "  Creating resource group..." -ForegroundColor Yellow
    az group create --name $resourceGroup --location $location | Out-Null
    Write-Host "  [OK] Resource group created" -ForegroundColor Green
}

# Check if VM already exists
Write-Host "`n[3/6] Checking if VM already exists..." -ForegroundColor Yellow
$vmExists = az vm show --resource-group $resourceGroup --name $vmName 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [WARNING] VM '$vmName' already exists" -ForegroundColor Yellow
    $response = Read-Host "  Do you want to delete and recreate it? (y/N)"
    if ($response -eq "y" -or $response -eq "Y") {
        Write-Host "  Deleting existing VM..." -ForegroundColor Yellow
        az vm delete --resource-group $resourceGroup --name $vmName --yes | Out-Null
        Write-Host "  [OK] VM deleted" -ForegroundColor Green
    } else {
        Write-Host "  [ABORTED] Keeping existing VM" -ForegroundColor Yellow
        exit 0
    }
} else {
    Write-Host "  [OK] VM name available" -ForegroundColor Green
}

# Create VM
Write-Host "`n[4/6] Creating VM (this takes 2-3 minutes)..." -ForegroundColor Yellow
Write-Host "  VM Name: $vmName" -ForegroundColor Gray
Write-Host "  Size: $vmSize (2 vCPU, 16GB RAM, 8GB SGX EPC)" -ForegroundColor Gray
Write-Host "  Location: $location" -ForegroundColor Gray
Write-Host "  Image: Ubuntu 22.04 LTS" -ForegroundColor Gray
Write-Host ""

# Create VM and capture output
$vmCreateResult = az vm create `
    --resource-group $resourceGroup `
    --name $vmName `
    --size $vmSize `
    --location $location `
    --image $ubuntuImage `
    --admin-username $adminUsername `
    --generate-ssh-keys `
    --public-ip-sku Standard `
    --output json 2>&1

if ($LASTEXITCODE -eq 0) {
    # Filter out non-JSON lines (warnings, etc) and parse JSON
    $jsonOutput = $vmCreateResult | Where-Object { $_ -match '^\s*[{\[]' -or $_ -match '^\s*["\d]' -or $_ -match '^\s*}' } | Out-String
    
    try {
        $vmInfo = $jsonOutput | ConvertFrom-Json
        Write-Host "  [OK] VM created successfully" -ForegroundColor Green
        Write-Host "  Public IP: $($vmInfo.publicIpAddress)" -ForegroundColor Cyan
        Write-Host "  Private IP: $($vmInfo.privateIpAddress)" -ForegroundColor Gray
        
        $publicIp = $vmInfo.publicIpAddress
    } catch {
        Write-Host "  [WARNING] VM created but failed to parse output" -ForegroundColor Yellow
        # Try to get IP using separate command
        Write-Host "  Fetching VM information..." -ForegroundColor Gray
        $publicIp = az vm show -d --resource-group $resourceGroup --name $vmName --query publicIps -o tsv
        Write-Host "  Public IP: $publicIp" -ForegroundColor Cyan
    }
} else {
    Write-Host "  [ERROR] Failed to create VM" -ForegroundColor Red
    Write-Host $vmCreateResult -ForegroundColor Red
    exit 1
}

# Install Docker via cloud-init
Write-Host "`n[5/6] Installing Docker on VM..." -ForegroundColor Yellow
Write-Host "  This may take 1-2 minutes..." -ForegroundColor Gray

$dockerInstallScript = @'
#!/bin/bash
set -e

# Update system
sudo apt-get update

# Install Docker
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Add user to docker group
sudo usermod -aG docker azureuser

# Verify installation
docker --version
'@

$tempScript = New-TemporaryFile
$dockerInstallScript | Out-File -FilePath $tempScript.FullName -Encoding UTF8

try {
    # Copy script to VM
    scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null `
        $tempScript.FullName "${adminUsername}@${publicIp}:/tmp/install_docker.sh" 2>&1 | Out-Null
    
    # Execute script on VM
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null `
        "${adminUsername}@${publicIp}" "chmod +x /tmp/install_docker.sh && /tmp/install_docker.sh" 2>&1 | Out-Null
    
    Write-Host "  [OK] Docker installed successfully" -ForegroundColor Green
} catch {
    Write-Host "  [WARNING] Automatic Docker installation failed" -ForegroundColor Yellow
    Write-Host "  You may need to install Docker manually after SSH" -ForegroundColor Gray
} finally {
    Remove-Item $tempScript.FullName -Force
}

# Summary
Write-Host "`n[6/6] VM Setup Complete!" -ForegroundColor Green
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "VM Information:" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Name:          $vmName" -ForegroundColor White
Write-Host "  Resource Group: $resourceGroup" -ForegroundColor White
Write-Host "  Size:          $vmSize" -ForegroundColor White
Write-Host "  Location:      $location" -ForegroundColor White
Write-Host "  Public IP:     $publicIp" -ForegroundColor Cyan
Write-Host "  Username:      $adminUsername" -ForegroundColor White
Write-Host ""
Write-Host "SSH Connection:" -ForegroundColor Cyan
Write-Host "  ssh ${adminUsername}@${publicIp}" -ForegroundColor Yellow
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Connect to VM:" -ForegroundColor White
Write-Host "     ssh ${adminUsername}@${publicIp}" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Upload benchmark files:" -ForegroundColor White
Write-Host "     scp tee-benchmark-*.tar.gz ${adminUsername}@${publicIp}:~/" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. On VM, extract and setup:" -ForegroundColor White
Write-Host "     tar xzf tee-benchmark-*.tar.gz" -ForegroundColor Gray
Write-Host "     mkdir -p results" -ForegroundColor Gray
Write-Host "     docker build -f Dockerfile.gramine -t tee-benchmark-gramine:latest ." -ForegroundColor Gray
Write-Host ""
Write-Host "  4. Run benchmark:" -ForegroundColor White
Write-Host "     chmod +x run_cloud_test.sh" -ForegroundColor Gray
Write-Host "     ./run_cloud_test.sh" -ForegroundColor Gray
Write-Host ""
Write-Host "  5. Download results:" -ForegroundColor White
Write-Host "     scp ${adminUsername}@${publicIp}:~/results/*.json ./results/" -ForegroundColor Gray
Write-Host ""
Write-Host "Cost Estimate:" -ForegroundColor Cyan
Write-Host "  DC2s_v3: ~`$0.248/hour" -ForegroundColor Gray
Write-Host "  Complete test: ~10 minutes = ~`$0.04" -ForegroundColor Gray
Write-Host ""
Write-Host "To delete VM when done:" -ForegroundColor Yellow
Write-Host "  az vm delete --resource-group $resourceGroup --name $vmName --yes" -ForegroundColor Gray
Write-Host "============================================================" -ForegroundColor Cyan
