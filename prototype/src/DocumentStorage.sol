// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title DocumentStorage
 * @notice Marketplace + provenance for Bio-Block IPFS documents.
 * @dev Document lists are NOT stored in contract state. Clients reconstruct a
 *      user's documents from indexed `DocumentStored` / `DocumentDeleted` logs.
 *      This is cheaper than maintaining `mapping(address => string[])`.
 *      Addresses: https://github.com/healthyinc/bio-block/issues/156
 *                 https://github.com/healthyinc/bio-block/issues/146
 */
contract DocumentStorage {
    mapping(string => uint256) public documentPrices;
    mapping(string => address) public documentOwners;
    mapping(address => uint256) public earnings;

    event DocumentStored(address indexed owner, string ipfsHash, uint256 price);
    event DocumentDeleted(address indexed owner, string ipfsHash);
    event DocumentPurchased(
        address indexed buyer,
        address indexed owner,
        string ipfsHash,
        uint256 amount
    );
    event EarningsWithdrawn(address indexed owner, uint256 amount);

    function storeDocument(string memory ipfsHash, uint256 price) public {
        require(bytes(ipfsHash).length > 0, "Empty IPFS hash");
        require(documentOwners[ipfsHash] == address(0), "Document already exists");

        documentPrices[ipfsHash] = price;
        documentOwners[ipfsHash] = msg.sender;

        emit DocumentStored(msg.sender, ipfsHash, price);
    }

    function purchaseDocument(string memory ipfsHash) public payable returns (bool) {
        address owner = documentOwners[ipfsHash];
        require(owner != address(0), "Document not found");
        require(msg.value >= documentPrices[ipfsHash], "Insufficient payment");

        earnings[owner] += msg.value;
        emit DocumentPurchased(msg.sender, owner, ipfsHash, msg.value);
        return true;
    }

    function withdrawEarnings() public {
        uint256 amount = earnings[msg.sender];
        require(amount > 0, "No earnings");
        earnings[msg.sender] = 0;
        payable(msg.sender).transfer(amount);
        emit EarningsWithdrawn(msg.sender, amount);
    }

    /**
     * @notice Clears on-chain ownership/price for a document owned by the caller.
     * @dev Emits `DocumentDeleted` so indexers/frontends can drop the hash from
     *      event-sourced document lists (no on-chain array to prune).
     */
    function deleteDocument(string memory ipfsHash) public {
        require(documentOwners[ipfsHash] == msg.sender, "Not document owner");

        delete documentPrices[ipfsHash];
        delete documentOwners[ipfsHash];

        emit DocumentDeleted(msg.sender, ipfsHash);
    }
}
