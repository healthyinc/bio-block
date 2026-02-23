// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DocumentStorage {
    mapping(address => string[]) private userDocuments;
    mapping(string => uint256) public documentPrices;
    mapping(string => address) public documentOwners;
    mapping(address => uint256) public earnings;
    
    function storeDocument(string memory ipfsHash, uint256 price) public {
        userDocuments[msg.sender].push(ipfsHash);
        documentPrices[ipfsHash] = price;
        documentOwners[ipfsHash] = msg.sender;
    }
    
    function purchaseDocument(string memory ipfsHash) public payable returns (bool) {
        require(documentOwners[ipfsHash] != address(0), "Document does not exist");
        require(msg.sender != documentOwners[ipfsHash], "Cannot purchase own document");
        uint256 price = documentPrices[ipfsHash];
        require(msg.value == price, "Exact price required");
        earnings[documentOwners[ipfsHash]] += price;
        return true;
    }
    
    function withdrawEarnings() public {
        uint256 amount = earnings[msg.sender];
        require(amount > 0, "No earnings");
        earnings[msg.sender] = 0;
        payable(msg.sender).transfer(amount);
    }
    
    function getDocuments(address user) public view returns (string[] memory) {
        return userDocuments[user];
    }
    
    function getMyDocuments() public view returns (string[] memory) {
        return userDocuments[msg.sender];
    }
}