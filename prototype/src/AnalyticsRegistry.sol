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

    /// @notice The relayer address that is authorized to submit registrations
    /// on behalf of analysts (meta-transaction / gasless pattern).
    address public relayer;

    mapping(address => AnalyticsRecord[]) private userAnalytics;
    mapping(bytes32 => string[]) private datasetResults;

    // Prevents the same result CID from being registered twice
    mapping(bytes32 => bool) private registeredResults;

    modifier onlyRelayer() {
        require(msg.sender == relayer, "Only relayer can register");
        _;
    }

    constructor() {
        relayer = msg.sender;
    }

    /// @notice Register an analytics result on-chain.
    /// @dev Only the relayer (server wallet) may call this function.
    ///      The `analyst` parameter is the EIP-712-authenticated wallet
    ///      of the user who actually performed the analysis, so the
    ///      on-chain audit trail matches the off-chain IPFS record.
    /// @param sourceCID  IPFS CID of the source dataset.
    /// @param resultCID  IPFS CID of the analytics result JSON.
    /// @param analysisType  Type of analysis (e.g. "descriptive").
    /// @param analyst  Ethereum address of the real analyst.
    function registerAnalytics(
        string memory sourceCID,
        string memory resultCID,
        string memory analysisType,
        address analyst
    ) public onlyRelayer {
        require(bytes(sourceCID).length > 0, "Invalid source CID");
        require(bytes(resultCID).length > 0, "Invalid result CID");
        require(bytes(analysisType).length > 0, "Invalid analysis type");
        require(analyst != address(0), "Invalid analyst address");

        bytes32 resultKey = keccak256(bytes(resultCID));
        require(!registeredResults[resultKey], "Result CID already registered");

        registeredResults[resultKey] = true;

        userAnalytics[analyst].push(AnalyticsRecord({
            sourceCID: sourceCID,
            resultCID: resultCID,
            analysisType: analysisType,
            timestamp: block.timestamp
        }));

        bytes32 datasetId = keccak256(bytes(sourceCID));
        datasetResults[datasetId].push(resultCID);

        emit AnalyticsRegistered(
            analyst,
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

    /// @notice Query analytics records for any analyst address.
    /// @dev Changed from msg.sender to explicit `analyst` param so the
    ///      server can relay queries on behalf of users.
    function getAnalyticsForAddress(
        address analyst,
        uint256 offset,
        uint256 limit
    ) public view returns (AnalyticsRecord[] memory) {
        AnalyticsRecord[] storage records = userAnalytics[analyst];
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

    /// @notice Return count for any analyst address.
    function getAnalyticsCount(address analyst) public view returns (uint256) {
        return userAnalytics[analyst].length;
    }
}
