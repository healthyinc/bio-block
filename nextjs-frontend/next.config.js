/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  eslint: {
    ignoreDuringBuilds: false,
  },
  // Suppress workspace root warning
  outputFileTracingRoot: require('path').join(__dirname, '..'),
  // Image optimization
  images: {
    formats: ['image/avif', 'image/webp'],
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**',
      },
    ],
  },
  // Compression
  compress: true,
  // Power optimizations
  poweredByHeader: false,
  // Webpack configuration for browser compatibility with crypto and blockchain libraries
  // These fallbacks are required because ethers.js and crypto-js use Node.js-specific modules
  // that don't exist in the browser environment
  webpack: (config, { isServer }) => {
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false, // File system access not available in browser
      net: false, // Network protocols handled by browser APIs
      tls: false, // TLS/SSL handled by browser
    };
    // Tree-shaking optimization
    if (!isServer) {
      config.optimization = {
        ...config.optimization,
        splitChunks: {
          ...config.optimization?.splitChunks,
          cacheGroups: {
            ...(typeof config.optimization?.splitChunks === 'object' 
              ? config.optimization.splitChunks.cacheGroups 
              : {}),
            vendor: {
              test: /[\\/]node_modules[\\/]/,
              name: 'vendors',
              chunks: 'all',
            },
          },
        },
      };
    }
    return config;
  },
};

module.exports = nextConfig;
