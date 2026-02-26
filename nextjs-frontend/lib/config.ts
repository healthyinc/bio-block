/**
 * Blockchain Configuration for BioBlock
 * 
 * This file contains blockchain and environment configuration.
 * Primary wallet configuration is now in wagmi.ts
 */

import { sepolia } from 'viem/chains';

// ==========================================
// CHAIN CONFIGURATION
// ==========================================

/**
 * Default chain for the application
 */
export const DEFAULT_CHAIN = sepolia;

// ==========================================
// CONTRACT CONFIGURATION
// ==========================================

/**
 * BioBlock Smart Contract Address
 * Note: Primary contract config is in wagmi.ts, this is kept for backwards compatibility
 */
export const CONTRACT_ADDRESS = '0xd58de64aac08d5412b8020c7c61b215fec0c9644' as const;

// ==========================================
// WALLET TYPES
// ==========================================

export enum WalletType {
  NONE = 'none',
  EOA = 'eoa', // External Owned Account (MetaMask, etc.)
  EMBEDDED = 'embedded', // Embedded wallet (deprecated - now using RainbowKit)
}

// ==========================================
// ENVIRONMENT HELPERS
// ==========================================

export const isProduction = process.env.NODE_ENV === 'production';
export const isDevelopment = process.env.NODE_ENV === 'development';
