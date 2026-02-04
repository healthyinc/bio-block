// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DocumentStorage {
    event WithdrawalSuccess(address indexed user, uint256 amount);
    
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
        require(msg.value >= documentPrices[ipfsHash], "Insufficient payment");
        earnings[documentOwners[ipfsHash]] += msg.value;
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
    
    function getDocuments(address user) public view returns (string[] memory) {
        return userDocuments[user];
    }
    
    function getMyDocuments() public view returns (string[] memory) {
        return userDocuments[msg.sender];
    }
}