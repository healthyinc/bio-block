import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Suppress workspace root warning
  outputFileTracingRoot: join(__dirname, '..'),
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
  // Security: remove X-Powered-By header
  poweredByHeader: false,
  // Turbopack is the default bundler in Next.js 16
  // It natively handles Node.js polyfill fallbacks (fs, net, tls)
  // and optimized chunk splitting — no custom webpack config needed.
  turbopack: {},
};

export default nextConfig;
