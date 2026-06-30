// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AnalyticsRegistry {
    struct AnalyticsRecord {
        string sourceCID;
        string resultCID;
        string analysisType;
        uint256 timestamp;
    }

    event AnalyticsRegistered(
        address indexed analyst,
        string sourceCID,
        string resultCID,
        string analysisType,
        uint256 timestamp
    );

    mapping(address => AnalyticsRecord[]) private userAnalytics;
    mapping(bytes32 => string[]) private datasetResults;

    // Prevents the same result CID from being registered twice
    mapping(bytes32 => bool) private registeredResults;

    function registerAnalytics(
        string memory sourceCID,
        string memory resultCID,
        string memory analysisType
    ) public {
        require(bytes(sourceCID).length > 0, "Invalid source CID");
        require(bytes(resultCID).length > 0, "Invalid result CID");
        require(bytes(analysisType).length > 0, "Invalid analysis type");

        bytes32 resultKey = keccak256(bytes(resultCID));
        require(!registeredResults[resultKey], "Result CID already registered");

        registeredResults[resultKey] = true;

        userAnalytics[msg.sender].push(AnalyticsRecord({
            sourceCID: sourceCID,
            resultCID: resultCID,
            analysisType: analysisType,
            timestamp: block.timestamp
        }));

        bytes32 datasetId = keccak256(bytes(sourceCID));
        datasetResults[datasetId].push(resultCID);

        emit AnalyticsRegistered(
            msg.sender,
            sourceCID,
            resultCID,
            analysisType,
            block.timestamp
        );
    }

    function getAnalyticsForDataset(string memory sourceCID) public view returns (string[] memory) {
        bytes32 datasetId = keccak256(bytes(sourceCID));
        return datasetResults[datasetId];
    }

    function getMyAnalytics(uint256 offset, uint256 limit) public view returns (AnalyticsRecord[] memory) {
        AnalyticsRecord[] storage records = userAnalytics[msg.sender];
        uint256 total = records.length;
        
        if (offset >= total) {
            return new AnalyticsRecord[](0);
        }
        
        uint256 end = offset + limit;
        if (end > total) {
            end = total;
        }
        
        uint256 size = end - offset;
        AnalyticsRecord[] memory result = new AnalyticsRecord[](size);
        
        for (uint256 i = 0; i < size; i++) {
            result[i] = records[offset + i];
        }
        
        return result;
    }

    function getMyAnalyticsCount() public view returns (uint256) {
        return userAnalytics[msg.sender].length;
    }
}
