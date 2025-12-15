# Account Roles Configuration

## 4-Account Setup for Complete Business Logic Testing

### Account Allocation

```
Account 1: Owner (Platform Administrator)
├─ Address: 0x8677244ad457159b988dbbc34c92900d16d57445
├─ Balance: 0.746 ETH (Sepolia testnet)
├─ Role: Contract deployer and platform manager
├─ Responsibilities:
│  ├─ Deploy DataEquityProtocol contract
│  ├─ Manage TEE signer whitelist (setTEESigner)
│  └─ Platform governance
└─ Represents: Platform operator (e.g., IoT data marketplace platform)

Account 2: Buyer (Data Purchaser)
├─ Address: 0x49556d7f15d92e68402d69046f2899a37b212589
├─ Balance: 0.524 ETH (Sepolia testnet)
├─ Role: Data consumer who purchases IoT data
├─ Responsibilities:
│  ├─ Create orders (createOrder)
│  ├─ Lock payment funds
│  └─ Submit TEE signatures for settlement
└─ Represents: AI/ML company needing training data (e.g., traffic prediction service)

Account 3: Seller (Data Provider)
├─ Address: 0x5b7f0f1a6261f6eb218c377fb59927ae9d3b0361
├─ Balance: 0 ETH (no ETH needed - receive-only)
├─ Role: IoT device owner providing sensor data
├─ Responsibilities:
│  ├─ Upload sensor data to TEE
│  └─ Receive payments based on data quality
└─ Represents: Individual/organization with IoT devices (e.g., car owners with sensors)

Account 4: TEE Signer (Trusted Execution Environment)
├─ Address: 0x5950e7ba9ff49a1b57ee2610212de2036dea5a9c
├─ Balance: 0 ETH (no ETH needed - off-chain signing only)
├─ Role: Independent trusted third party for data quality validation
├─ Responsibilities:
│  ├─ Evaluate data quality in secure enclave (Intel SGX/AMD SEV)
│  ├─ Calculate utility score (u)
│  └─ Sign (orderId, utility, nonce) with private key
└─ Represents: Trusted computing node or audit organization
```

## Business Flow with 4 Separate Accounts

### Phase 1: Platform Setup (Owner)
```
Owner → Deploy Contract
  ├─ Set initial TEE signer address (Account 4)
  └─ Contract address: 0x...
```

### Phase 2: Order Creation (Buyer)
```
Buyer → Create Order
  ├─ Specify Seller address (Account 3)
  ├─ Set pricing parameters (p_base, α, k)
  ├─ Lock funds: e.g., 0.1 ETH
  └─ Set deadline: 3600 seconds
```

### Phase 3: Data Quality Evaluation (Seller + TEE)
```
Seller → Upload Data → TEE
  ├─ IoT sensor data sent to TEE enclave
  └─ TEE evaluates with LSTM model
  
TEE → Generate Signature
  ├─ Calculate utility score: u = 0.95
  ├─ Sign: sig = sign(orderId, u, nonce)
  └─ Return signature to Buyer
```

### Phase 4: Settlement (Buyer triggers, Seller receives)
```
Buyer → Submit Settlement
  ├─ Call settleTransaction(orderId, utility, sig)
  └─ Contract verifies TEE signature
  
Smart Contract → Execute Payment
  ├─ Calculate: finalPrice = p_base + α * k * u
  ├─ Transfer to Seller: 0.0X ETH
  └─ Refund excess to Buyer
```

## Why 4 Separate Accounts?

### Decentralization
- Owner ≠ Buyer: Platform doesn't control trades
- Independent roles prevent conflicts of interest

### Trust Model
- TEE Signer is neutral third party
- Can't be controlled by Buyer or Seller
- Cryptographic signature ensures authenticity

### Real-world Mapping
- Matches actual IoT data marketplace architecture
- Clear separation of concerns
- Each entity has distinct economic incentives

## Testing Strategy

### Scenario 1: Happy Path
```
Owner deploys → Buyer creates order → Seller provides data 
→ TEE validates (u=0.95) → Settlement succeeds
```

### Scenario 2: Low Quality Data
```
Same flow but TEE returns u=0.3 
→ Seller receives reduced payment
```

### Scenario 3: Timeout Protection
```
Seller doesn't provide data → Deadline expires 
→ Buyer calls refund() → Full refund
```

### Scenario 4: Invalid Signature Attack
```
Malicious actor tries fake signature 
→ Contract rejects (ECDSA verification fails)
```

## Account Funding Requirements

| Account | Needs ETH? | Amount Needed | Purpose |
|---------|-----------|---------------|---------|
| Owner (1) | ✅ Yes | ~0.05 ETH | Contract deployment gas |
| Buyer (2) | ✅ Yes | ~0.2 ETH | Order payment + gas |
| Seller (3) | ❌ No | 0 ETH | Receive-only |
| TEE (4) | ❌ No | 0 ETH | Off-chain signing |

**Total:** ~0.25 ETH recommended for full testing
**Current:** 1.27 ETH available ✅ Sufficient

## Configuration Commands

```bash
# Check all balances
npx hardhat run scripts/check-all-balances.js --network sepolia

# Deploy as Owner (Account 1)
npx hardhat ignition deploy ignition/modules/DataEquityProtocol.ts --network sepolia

# Run full test suite with all 4 accounts
npx hardhat test --network sepolia

# Run benchmark tests
npx hardhat test test/DataEquityProtocol.benchmark.js --network sepolia
```

## Security Notes

- Private keys stored securely in Hardhat keystore (encrypted)
- TEE Signer private key should ideally be in hardware security module (HSM)
- In production, use multi-sig wallet for Owner
- Never commit `.env` or keystore files to git
