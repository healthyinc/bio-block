require("@nomiclabs/hardhat-ethers");
require("@nomiclabs/hardhat-waffle");

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: "0.8.0",
  paths: {
    sources: "./src",
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts"
  }
};