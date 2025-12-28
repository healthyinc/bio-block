/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  eslint: {
    ignoreDuringBuilds: false,
  },
  // Webpack configuration for browser compatibility with crypto and blockchain libraries
  // These fallbacks are required because ethers.js and crypto-js use Node.js-specific modules
  // that don't exist in the browser environment
  webpack: (config) => {
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false, // File system access not available in browser
      net: false, // Network protocols handled by browser APIs
      tls: false, // TLS/SSL handled by browser
    };
    return config;
  },
};

module.exports = nextConfig;
