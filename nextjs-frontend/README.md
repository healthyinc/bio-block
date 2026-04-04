# Bio-Block Next.js Frontend

Next.js 15 + TypeScript frontend for Bio-Block — a decentralized medical records platform. Migrated from the React prototype (`prototype/`) with full type safety, App Router, and API route proxying.

## Architecture

```
Browser                        Backends                    Storage
───────                        ────────                    ───────
User File                      JS Backend (:3001)          IPFS (Pinata)
  → Anonymize (via backend)    Python Backend (:3002)      Ethereum Blockchain
  → Encrypt (client-side)           │                         │
  → Upload encrypted blob ─────────┘─── IPFS ────────────────┘
  → Store hash on-chain ──────────────── Blockchain ──────────┘
```

All encryption happens in the browser. Backends handle IPFS pinning, metadata storage, anonymization, and search.

## Prerequisites

- Node.js 18+
- MetaMask browser extension
- JavaScript backend running on `http://localhost:3001`
- Python backend running on `http://localhost:3002`

## Quick Start

```bash
# Install dependencies
npm install

# Copy environment template
cp .env.example .env.local

# Generate a 32-byte hex encryption key
openssl rand -hex 32
# Paste it as NEXT_PUBLIC_ENCRYPTION_KEY in .env.local

# Start dev server
npm run dev
```

Open http://localhost:3000

## Running All Services

```bash
# Terminal 1 — JavaScript Backend (IPFS, anonymization)
cd ../javascript_backend
npm install && npm start        # http://localhost:3001

# Terminal 2 — Python Backend (search, metadata, advanced anonymization)
cd ../python_backend
pip install -r requirements.txt
python main.py                  # http://localhost:3002

# Terminal 3 — Next.js Frontend
cd ../nextjs-frontend
npm run dev                     # http://localhost:3000
```

## Project Structure

```
nextjs-frontend/
├── app/
│   ├── layout.tsx              # Root layout with metadata
│   ├── page.tsx                # Homepage — wallet connect, navigation
│   ├── globals.css             # Global styles (Tailwind base)
│   ├── actions.ts              # Server Actions (search, store, health)
│   ├── api/
│   │   ├── health/route.ts     # GET  /api/health
│   │   ├── search/route.ts     # POST /api/search
│   │   ├── store/route.ts      # POST /api/store
│   │   ├── anonymize/route.ts  # POST /api/anonymize
│   │   ├── ipfs/upload/route.ts# POST /api/ipfs/upload
│   │   ├── preview/[type]/route.ts # POST /api/preview/:type
│   │   └── proxy/route.ts      # General proxy
│   ├── dashboard/
│   │   ├── layout.tsx
│   │   └── page.tsx            # Earnings, documents, withdraw
│   ├── search/
│   │   ├── layout.tsx
│   │   └── page.tsx            # Search with filters, preview modal
│   └── upload/
│       ├── layout.tsx
│       └── page.tsx            # Upload with anonymization pipeline
├── components/
│   ├── Dashboard.tsx           # Earnings + document list
│   ├── SearchData.tsx          # Search UI, filters, purchase & download
│   ├── UploadData.tsx          # Upload form, encryption, IPFS pipeline
│   ├── Header.tsx              # Wallet dropdown header
│   ├── HeroSection.tsx         # Landing hero section
│   └── FeatureCard.tsx         # Feature card component
├── lib/
│   ├── api.ts                  # Typed API client helpers
│   ├── contractService.ts      # Ethers.js v6 contract interactions
│   ├── encryptionUtils.ts      # AES-256-CBC encrypt/decrypt
│   ├── streamingEncryption.ts  # Chunked encryption for large files
│   └── types.ts                # Shared TypeScript interfaces
├── types/
│   └── ethereum.d.ts           # Window.ethereum type declarations
├── next.config.js              # Image optimization, compression, splitting
├── tailwind.config.js          # Tailwind content paths (.ts/.tsx)
├── tsconfig.json               # ES2018 target, strict mode
└── package.json
```

## Environment Variables

Create `.env.local`:

```env
# Required
NEXT_PUBLIC_ENCRYPTION_KEY=<32-byte-hex-key>

# Backend URLs (defaults shown)
NEXT_PUBLIC_JS_BACKEND_URL=http://localhost:3001
NEXT_PUBLIC_PYTHON_BACKEND_URL=http://localhost:3002

# Optional — production blockchain
# NEXT_PUBLIC_RPC_URL=https://sepolia.infura.io/v3/YOUR_PROJECT_ID
# NEXT_PUBLIC_CONTRACT_ADDRESS=0x...
```

## Production Build

```bash
npm run build    # Compiles, lints, type-checks, generates static pages
npm start        # Starts production server
```

Build output:

- 4 static pages (`/`, `/dashboard`, `/search`, `/upload`)
- 7 dynamic API routes
- ~279 KB shared JS bundle

## Features

### Wallet Connection

Connect MetaMask, view address in header dropdown, disconnect anytime.

### Upload Documents

Select file (Excel, CSV, Image, PDF, DICOM) → fill metadata (title, description, disease tags, data type, gender, source, price) → automatic anonymization → client-side encryption → IPFS upload → blockchain hash storage → metadata saved to Python backend.

### Search Documents

Enter query or use advanced filters (data type, gender, data source, file type) → results from Python backend → preview 5% sample of Excel data → purchase with ETH → download & decrypt.

### Dashboard

View earnings balance, withdraw ETH, list uploaded documents with prices, download own documents.

## Tech Stack

| Layer      | Technology                  |
| ---------- | --------------------------- |
| Framework  | Next.js 15 (App Router)     |
| Language   | TypeScript (ES2018)         |
| Styling    | Tailwind CSS                |
| Icons      | Lucide React                |
| Blockchain | Ethers.js v6                |
| Encryption | CryptoJS (AES-256-CBC)      |
| IPFS       | Pinata via JS backend       |
| Search     | ChromaDB via Python backend |

## Troubleshooting

**Port in use:**

```bash
npx kill-port 3000
```

**Encryption key error:** Generate a new key with `openssl rand -hex 32` and update `.env.local`.

**Backend not connecting:** Ensure JS backend (`:3001`) and Python backend (`:3002`) are running. Check CORS settings.

**MetaMask issues:** Install extension, connect to Sepolia testnet, ensure you have test ETH.

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md).

## License

See [LICENSE](../LICENSE).
