import { buildModule } from "@nomicfoundation/hardhat-ignition/modules";

const DataEquityProtocolModule = buildModule("DataEquityProtocolModule", (m) => {
  // Get the deployer account as the TEE signer
  const teeSigner = m.getAccount(0);

  // Deploy the DataEquityProtocol contract
  const dataEquityProtocol = m.contract("DataEquityProtocol", [teeSigner]);

  return { dataEquityProtocol };
});

export default DataEquityProtocolModule;
