# Security Guidelines

## 🔒 Sensitive Information Protection

### Never Commit These Files:
- `.env` - Contains private keys and RPC URLs
- `*.key`, `*.pem` - Cryptographic keys
- Any file with actual private keys or passwords

### What's Safe to Share:
✅ Public wallet addresses (in ACCOUNTS.md) - These are testnet addresses and publicly visible on blockchain explorers
✅ Smart contract addresses - Publicly deployed contracts
✅ RPC URLs for public testnets (Sepolia, Arbitrum Sepolia, etc.)
✅ Configuration templates (`.env.example`)

## 🔑 Private Key Management

### For Development/Testing:
1. **Use `.env` file** (never commit it!)
   ```bash
   cp .env.example .env
   # Fill in your actual keys
   ```

2. **Testnet Only**: The accounts in this project are for testnet use only
   - Owner: Platform admin account
   - Buyer: Data purchaser account  
   - Seller: Data provider account
   - TEE Signer: Trusted computation signer

3. **Get Testnet ETH**:
   - Sepolia: https://sepoliafaucet.com/
   - Arbitrum Sepolia: https://faucet.quicknode.com/arbitrum/sepolia
   - Optimism Sepolia: https://app.optimism.io/faucet

### For Production:
⚠️ **NEVER use testnet keys in production!**

1. Use hardware wallets (Ledger/Trezor)
2. Use environment variables on secure servers
3. Use Azure Key Vault / AWS Secrets Manager for cloud deployments
4. Enable multi-sig for critical contracts

## 🌐 Azure/Cloud Security

### TEE Deployment:
The `tee/create_azure_vm.ps1` script contains:
- Resource group name (default: "web3") - You should change this
- No hardcoded credentials - Uses `az login` for authentication
- SSH keys generated automatically via `--generate-ssh-keys`

### Recommended Changes Before Use:
```powershell
# In tee/create_azure_vm.ps1, change:
$resourceGroup = "your-unique-resource-group-name"  # Not "web3"
$vmName = "your-vm-name"                            # Not default
```

## 📊 Data Privacy

### Large Files Not Included:
Due to size constraints, these files are in `.gitignore`:
- `input/pems-dataset/data/*.npz` (~24MB)
- `tee/data/PEMS08.npz` (~24MB)
- `tee/models/traffic_lstm.pth` (~15MB)

**Download instructions**: See [README.md](README.md#quick-start)

### PeMS Dataset:
- Public traffic dataset from Caltrans
- No personal information
- Used for research purposes under their terms

## 🐛 Vulnerability Reporting

If you discover a security vulnerability:
1. **DO NOT** open a public GitHub issue
2. Email: [Your contact email]
3. Include:
   - Description of vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## ✅ Security Checklist Before GitHub Upload

- [ ] No `.env` file in repo
- [ ] All private keys removed from code
- [ ] No API keys or passwords in code
- [ ] `.gitignore` properly configured
- [ ] Azure resource names changed from defaults
- [ ] README.md has download instructions for large files
- [ ] SECURITY.md reviewed and updated

## 📚 Additional Resources

- [Smart Contract Security Best Practices](https://consensys.github.io/smart-contract-best-practices/)
- [Azure Security Documentation](https://docs.microsoft.com/en-us/azure/security/)
- [Intel SGX Security Advisories](https://www.intel.com/content/www/us/en/security-center/default.html)
