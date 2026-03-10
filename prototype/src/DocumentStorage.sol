// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DocumentStorage {
    event DocumentStored(address indexed owner, string ipfsHash, uint256 price);
    event DocumentDeleted(address indexed owner, string ipfsHash);
    event MetadataUpdated(address indexed owner, string ipfsHash, uint256 newPrice, string newMetadata);
    event DocumentPurchased(address indexed buyer, address indexed seller, string ipfsHash, uint256 amount);

    event WithdrawalSuccess(address indexed user, uint256 amount);
    
    mapping(address => string[]) private userDocuments;
    mapping(bytes32 => uint256) public documentPrices;
    mapping(bytes32 => string) public documentMetadata;
    mapping(bytes32 => address) public documentOwners;
    mapping(address => uint256) public earnings;
    mapping(string => mapping(address => bool)) public hasPurchased;
    
    function storeDocument(string memory ipfsHash, uint256 price, string memory metadata) public {
        require(bytes(ipfsHash).length > 0, "Invalid IPFS hash");
        bytes32 docId = keccak256(bytes(ipfsHash));
        require(documentOwners[docId] == address(0), "Document already exists");
        require(price > 0, "Price must be greater than 0");  //price check
        userDocuments[msg.sender].push(ipfsHash);
        documentPrices[docId] = price;
        documentOwners[docId] = msg.sender;
        documentMetadata[docId] = metadata;
        emit DocumentStored(msg.sender, ipfsHash, price);
    }
    
    function updateMetadata(string memory ipfsHash, uint256 newPrice, string memory newMetadata) public {
        bytes32 docId = keccak256(bytes(ipfsHash));
        require(newPrice > 0, "Price must be greater than 0");
        require(documentOwners[docId] != address(0), "Document does not exist");
        require(documentOwners[docId] == msg.sender, "Only owner can update");
        documentPrices[docId] = newPrice;
        documentMetadata[docId] = newMetadata;
        emit MetadataUpdated(msg.sender, ipfsHash, newPrice, newMetadata);
    }

    function deleteDocument(string memory ipfsHash) public {
        bytes32 docId = keccak256(bytes(ipfsHash));
        require(documentOwners[docId] != address(0), "Document does not exist");
        require(documentOwners[docId] == msg.sender, "Only owner can delete");

        // Remove from userDocuments array
        string[] storage docs = userDocuments[msg.sender];
        for (uint256 i = 0; i < docs.length; i++) {
            if (keccak256(bytes(docs[i])) == keccak256(bytes(ipfsHash))) {
                docs[i] = docs[docs.length - 1];
                docs.pop();
                break;
            }
        }

        documentOwners[docId] = address(0);
        documentPrices[docId] = 0;
        documentMetadata[docId] = "";
        emit DocumentDeleted(msg.sender, ipfsHash);
    }

    function purchaseDocument(string memory ipfsHash) public payable returns (bool) {
        bytes32 docId = keccak256(bytes(ipfsHash));
        address owner = documentOwners[docId];
        require(owner != address(0), "Document does not exist");
        require(msg.sender != owner, "Cannot purchase own document");

        uint256 price = documentPrices[docId];
        require(msg.value == price, "Exact price required");

        earnings[owner] += price;
        hasPurchased[ipfsHash][msg.sender] = true;
        emit DocumentPurchased(msg.sender, owner, ipfsHash, price);
        return true;
    }
    
    function withdrawEarnings() public {
        uint256 amount = earnings[msg.sender];
        require(amount > 0, "No earnings");
        earnings[msg.sender] = 0;
         // Interaction after state update
        (bool sent, ) = payable(msg.sender).call{value: amount}("");
        require(sent, "Transfer failed");
        emit WithdrawalSuccess(msg.sender, amount);
    }
    
     function getMetadata(string memory ipfsHash) public view returns (string memory) {
        bytes32 docId = keccak256(bytes(ipfsHash));
        return documentMetadata[docId];
    }

    function hasAccess(string memory ipfsHash, address user) public view returns (bool) {
        return documentOwners[ipfsHash] == user || hasPurchased[ipfsHash][user];
    }

    function getDocuments(address user) public view returns (string[] memory) {
        return userDocuments[user];
    }
    
    function getMyDocuments() public view returns (string[] memory) {
        return userDocuments[msg.sender];
    }
}