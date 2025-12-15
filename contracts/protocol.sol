// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/cryptography/MessageHashUtils.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title DataEquityProtocol
 * @dev Implementation of the Risk-Hedging Equity Protocol for IoT Data Trading.
 * Based on the paper "Trustworthy Data Equity".
 */
contract DataEquityProtocol is ReentrancyGuard, Ownable {
    using ECDSA for bytes32;
    using MessageHashUtils for bytes32;

    // ============================
    // State Variables
    // ============================

    // The authorized TEE Enclave public key address
    address public teeSigner;

    // Fixed-point scaling factor (10^18) to handle decimals like alpha=0.5 or u=0.95
    uint256 public constant SCALE_FACTOR = 1e18;

    enum OrderState { CREATED, LOCKED, SETTLED, REFUNDED }

    struct Order {
        uint256 id;
        address buyer;
        address payable seller;
        uint256 p_base;      // Base fee (in Wei)
        uint256 alpha;       // Equity share (scaled by 1e18, e.g., 0.5 * 1e18)
        uint256 k_factor;    // Utility-to-Money coefficient (Wei per Utility Unit)
        uint256 maxDeposit;  // Maximum locked funds
        uint256 deadline;    // Timestamp for refund timeout
        bytes32 nonce;       // Unique nonce to prevent replay attacks
        OrderState state;
    }

    // Mapping from Order ID to Order details
    mapping(uint256 => Order) public orders;
    uint256 public orderCounter;

    // ============================
    // Events
    // ============================

    event OrderCreated(uint256 indexed orderId, address indexed buyer, address indexed seller, uint256 maxDeposit);
    event SettlementEvent(uint256 indexed orderId, uint256 utilityScore, uint256 finalPayment);
    event OrderRefunded(uint256 indexed orderId, uint256 amount);
    event TEESignerUpdated(address newSigner);

    // ============================
    // Constructor & Admin
    // ============================

    constructor(address _teeSigner) Ownable(msg.sender) {
        require(_teeSigner != address(0), "Invalid TEE signer");
        teeSigner = _teeSigner;
    }

    function setTEESigner(address _newSigner) external onlyOwner {
        require(_newSigner != address(0), "Invalid address");
        teeSigner = _newSigner;
        emit TEESignerUpdated(_newSigner);
    }

    // ============================
    // Phase 1: Order Creation & Locking
    // ============================

    /**
     * @notice Buyer creates an order and locks funds based on max potential liability.
     * @param _seller The data seller address.
     * @param _p_base The fixed base fee.
     * @param _alpha The equity share (scaled by 1e18).
     * @param _k_factor The coefficient converting utility to Wei.
     * @param _nonce Unique nonce generated off-chain.
     * @param _durationSeconds Validity period before refund is allowed.
     */
    function createOrder(
        address payable _seller,
        uint256 _p_base,
        uint256 _alpha,
        uint256 _k_factor,
        bytes32 _nonce,
        uint256 _durationSeconds
    ) external payable nonReentrant {
        require(_seller != address(0), "Invalid seller");
        require(_alpha <= SCALE_FACTOR, "Alpha cannot exceed 1.0");
        require(msg.value > 0, "Deposit required");

        // Per paper: Buyer deposits max potential payout to ensure atomicity.
        // We assume msg.value is the calculated Max Deposit.
        
        orderCounter++;
        uint256 newOrderId = orderCounter;

        orders[newOrderId] = Order({
            id: newOrderId,
            buyer: msg.sender,
            seller: _seller,
            p_base: _p_base,
            alpha: _alpha,
            k_factor: _k_factor,
            maxDeposit: msg.value,
            deadline: block.timestamp + _durationSeconds,
            nonce: _nonce,
            state: OrderState.LOCKED // Directly Locked as funds are escrowed
        });

        emit OrderCreated(newOrderId, msg.sender, _seller, msg.value);
    }

    // ============================
    // Phase 4: Atomic Settlement
    // ============================

    /**
     * @notice Algorithm 1: On-Chain Settlement Logic
     * @dev Verifies TEE signature and executes the equity pricing formula.
     * @param _orderId The ID of the order to settle.
     * @param _utility The realized utility score u (scaled by 1e18), calculated by TEE.
     * @param _signature The ECDSA signature provided by the TEE.
     */
    function settleTransaction(
        uint256 _orderId,
        uint256 _utility,
        bytes calldata _signature
    ) external nonReentrant {
        Order storage order = orders[_orderId];

        // 1. Check State
        require(order.state == OrderState.LOCKED, "Order not in LOCKED state");
        
        // 2. Verify Signature (Integrity & Correctness) 
        // Message format: Hash(orderId, utility, nonce)
        bytes32 messageHash = keccak256(abi.encodePacked(_orderId, _utility, order.nonce));
        bytes32 ethSignedMessageHash = messageHash.toEthSignedMessageHash();
        
        address recoveredSigner = ethSignedMessageHash.recover(_signature);
        require(recoveredSigner == teeSigner, "Invalid TEE signature");

        // 3. Execute Equity Pricing Formula [cite: 216, 397]
        // Payment = p + alpha * k * u
        // Note: alpha, k, and u are all scaled by 1e18.
        // Formula in code: (alpha * k * u) / SCALE_FACTOR / SCALE_FACTOR
        
        uint256 bonus = (order.alpha * order.k_factor * _utility) / SCALE_FACTOR / SCALE_FACTOR;
        uint256 finalPayment = order.p_base + bonus;

        // 4. Cap payment at Max Deposit (Safety check) [cite: 402]
        if (finalPayment > order.maxDeposit) {
            finalPayment = order.maxDeposit;
        }

        // 5. Update State
        order.state = OrderState.SETTLED;

        // 6. Transfer Funds (Atomic Settlement) [cite: 408-413]
        // Transfer payment to Seller
        (bool sentSeller, ) = order.seller.call{value: finalPayment}("");
        require(sentSeller, "Failed to send to Seller");

        // Refund remaining deposit to Buyer
        uint256 refundAmount = order.maxDeposit - finalPayment;
        if (refundAmount > 0) {
            (bool sentBuyer, ) = order.buyer.call{value: refundAmount}("");
            require(sentBuyer, "Failed to refund Buyer");
        }

        emit SettlementEvent(_orderId, _utility, finalPayment);
    }

    // ============================
    // Fallback: Refund Logic
    // ============================

    /**
     * @notice Allows buyer to claim refund if TEE fails to respond within deadline.
     * @param _orderId The order ID.
     */
    function refundOrder(uint256 _orderId) external nonReentrant {
        Order storage order = orders[_orderId];
        
        require(msg.sender == order.buyer, "Only buyer can refund");
        require(order.state == OrderState.LOCKED, "Invalid state");
        require(block.timestamp > order.deadline, "Deadline not passed");

        order.state = OrderState.REFUNDED;
        
        // Return full deposit to buyer
        (bool success, ) = order.buyer.call{value: order.maxDeposit}("");
        require(success, "Refund failed");

        emit OrderRefunded(_orderId, order.maxDeposit);
    }
}