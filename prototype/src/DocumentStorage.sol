// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DocumentStorage {
    event DocumentStored(address indexed owner, string ipfsHash, uint256 price);
    event DocumentDeleted(address indexed owner, string ipfsHash);
    event MetadataUpdated(address indexed owner, string ipfsHash, uint256 newPrice, string newMetadata);
    event DocumentPurchased(address indexed buyer, address indexed seller, string ipfsHash, uint256 amount);

    mapping(address => string[]) private userDocuments;
    mapping(string => uint256) public documentPrices;
    mapping(string => string) public documentMetadata;
    mapping(string => address) public documentOwners;
    mapping(address => uint256) public earnings;
    
    function storeDocument(string memory ipfsHash, uint256 price, string memory metadata) public {
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
        documentOwners[ipfsHash] = address(0);
        documentPrices[ipfsHash] = 0;
        documentMetadata[ipfsHash] = "";
        emit DocumentDeleted(msg.sender, ipfsHash);
    }

    function purchaseDocument(string memory ipfsHash) public payable returns (bool) {
        require(msg.value >= documentPrices[ipfsHash], "Insufficient payment");
        require(documentOwners[ipfsHash] != address(0), "Document does not exist");
        earnings[documentOwners[ipfsHash]] += msg.value;
        emit DocumentPurchased(msg.sender, documentOwners[ipfsHash], ipfsHash, msg.value);
        return true;
    }
    
    function withdrawEarnings() public {
        uint256 amount = earnings[msg.sender];
        require(amount > 0, "No earnings");
        earnings[msg.sender] = 0;
        payable(msg.sender).transfer(amount);
    }
    
     function getMetadata(string memory ipfsHash) public view returns (string memory) {
        return documentMetadata[ipfsHash];
    }

    function getDocuments(address user) public view returns (string[] memory) {
        return userDocuments[user];
    }
    
    function getMyDocuments() public view returns (string[] memory) {
        return userDocuments[msg.sender];
    }
}