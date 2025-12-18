# Bio-Block Next.js Frontend

Next.js 15 implementation of Bio-Block's hybrid architecture for decentralized medical records.

## Architecture

```
User File → Browser Encryption → Backend IPFS → Blockchain Storage
```

## Features

- Next.js 15 with App Router
- Client-side encryption (browser-based)
- MetaMask wallet integration
- Hybrid IPFS architecture
- Tailwind CSS styling

## Prerequisites

- Node.js 18+
- MetaMask browser extension
- Running JavaScript backend (see `../javascript_backend`)

## Quick Start

```bash
npm install
cp .env.example .env.local
# Edit .env.local with your actual values
npm run dev
```

Generate encryption key:
```bash
openssl rand -hex 32
```

## Development

**Important:** Start backend first to verify connectivity.

```bash
# Terminal 1 - Start backend
cd ../javascript_backend
npm start

# Terminal 2 - Start frontend
npm run dev
```

Visit http://localhost:3000

The frontend will automatically check backend connectivity and display status at the top of the page.

## Project Structure

```
nextjs-frontend/
├── app/
│   ├── layout.js
│   ├── page.js
│   └── globals.css
├── components/
│   ├── Header.js
│   ├── HeroSection.js
│   └── FeatureCard.js
├── next.config.js
└── tailwind.config.js
```