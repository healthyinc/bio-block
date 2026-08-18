// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DocumentStorage {
    mapping(address => string[]) private userDocuments;
    mapping(string => uint256) public documentPrices;
    mapping(string => address) public documentOwners;
    mapping(address => uint256) public earnings;
    mapping(string => mapping(address => bool)) public hasAccess;
    
    function storeDocument(string memory ipfsHash, uint256 price) public {
        userDocuments[msg.sender].push(ipfsHash);
        documentPrices[ipfsHash] = price;
        documentOwners[ipfsHash] = msg.sender;
        hasAccess[ipfsHash][msg.sender] = true;
    }
    
    function purchaseDocument(string memory ipfsHash) public payable returns (bool) {
        require(msg.value >= documentPrices[ipfsHash], "Insufficient payment");
        earnings[documentOwners[ipfsHash]] += msg.value;
        hasAccess[ipfsHash][msg.sender] = true;
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
    
    function checkAccess(string memory ipfsHash, address user) public view returns (bool) {
        return hasAccess[ipfsHash][user];
    }
}