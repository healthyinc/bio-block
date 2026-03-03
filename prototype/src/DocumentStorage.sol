// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DocumentStorage {
    event DocumentStored(address indexed owner, string ipfsHash, uint256 price);
    event DocumentDeleted(address indexed owner, string ipfsHash);
    event MetadataUpdated(address indexed owner, string ipfsHash, uint256 newPrice, string newMetadata);
    event DocumentPurchased(address indexed buyer, address indexed seller, string ipfsHash, uint256 amount);

    event WithdrawalSuccess(address indexed user, uint256 amount);
    
    mapping(address => string[]) private userDocuments;
    mapping(string => uint256) public documentPrices;
    mapping(string => string) public documentMetadata;
    mapping(string => address) public documentOwners;
    mapping(address => uint256) public earnings;
    mapping(string => mapping(address => bool)) public hasPurchased;
    
    function storeDocument(string memory ipfsHash, uint256 price, string memory metadata) public {
        require(documentOwners[ipfsHash] == address(0), "Document already exists");
        require(price > 0, "Price must be greater than 0");  //price check
        userDocuments[msg.sender].push(ipfsHash);
        documentPrices[ipfsHash] = price;
        documentOwners[ipfsHash] = msg.sender;
        documentMetadata[ipfsHash] = metadata;
        emit DocumentStored(msg.sender, ipfsHash, price);
    }
    
    function updateMetadata(string memory ipfsHash, uint256 newPrice, string memory newMetadata) public {
        require(documentOwners[ipfsHash] != address(0), "Document does not exist");
        require(documentOwners[ipfsHash] == msg.sender, "Only owner can update");
        documentPrices[ipfsHash] = newPrice;
        documentMetadata[ipfsHash] = newMetadata;
        emit MetadataUpdated(msg.sender, ipfsHash, newPrice, newMetadata);
    }

    function deleteDocument(string memory ipfsHash) public {
        require(documentOwners[ipfsHash] != address(0), "Document does not exist");
        require(documentOwners[ipfsHash] == msg.sender, "Only owner can delete");

        // Remove from userDocuments array
        string[] storage docs = userDocuments[msg.sender];
        for (uint256 i = 0; i < docs.length; i++) {
            if (keccak256(bytes(docs[i])) == keccak256(bytes(ipfsHash))) {
                docs[i] = docs[docs.length - 1];
                docs.pop();
                break;
            }
        }

        documentOwners[ipfsHash] = address(0);
        documentPrices[ipfsHash] = 0;
        documentMetadata[ipfsHash] = "";
        emit DocumentDeleted(msg.sender, ipfsHash);
    }

    function purchaseDocument(string memory ipfsHash) public payable returns (bool) {
        address owner = documentOwners[ipfsHash];
        require(owner != address(0), "Document does not exist");
        require(msg.sender != owner, "Cannot purchase own document");

        uint256 price = documentPrices[ipfsHash];
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
        return documentMetadata[ipfsHash];
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