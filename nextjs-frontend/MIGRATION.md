# Migration Guide: React Prototype to Next.js 15 + TypeScript

> Reference for reviewers and maintainers. Covers what changed, why, and how files map between the old React prototype and the new Next.js frontend.

## Summary

The React prototype (`prototype/src/`) was migrated to Next.js 15 with App Router and full TypeScript. All functionality is preserved — encryption, IPFS uploads, blockchain storage, wallet integration, search, anonymization.

**No breaking changes.** The prototype remains at `prototype/` for reference.

---

## File Mapping

### Components

| React (`prototype/src/`) | Next.js (`components/`) | Notes |
|--------------------------|------------------------|-------|
| `Dashboard.js` | `Dashboard.tsx` | Added typed props, interfaces |
| `search_data.js` | `SearchData.tsx` | Added filter interfaces, preview types |
| `upload_data.js` | `UploadData.tsx` | Added upload pipeline types |
| `App.js` (header section) | `Header.tsx` | Extracted as standalone component |
| `App.js` (hero section) | `HeroSection.tsx` | Extracted as standalone component |
| — | `FeatureCard.tsx` | New reusable card component |

### Utilities

| React (`prototype/src/`) | Next.js (`lib/`) | Notes |
|--------------------------|-----------------|-------|
| `contractService.js` | `contractService.ts` | Typed contract methods |
| `encryptionUtils.js` | `encryptionUtils.ts` | Typed encrypt/decrypt |
| `utils/streamingEncryption.js` | `streamingEncryption.ts` | Typed streaming class |
| — | `api.ts` | New typed API client helpers |
| — | `types.ts` | Shared interfaces |

### Pages (React Router to App Router)

| React | Next.js | Route |
|-------|---------|-------|
| `App.js` (conditional render) | `app/page.tsx` | `/` |
| `App.js` → `<Dashboard />` | `app/dashboard/page.tsx` | `/dashboard` |
| `App.js` → `<SearchData />` | `app/search/page.tsx` | `/search` |
| `App.js` → `<UploadData />` | `app/upload/page.tsx` | `/upload` |

### New Files (not in prototype)

| File | Purpose |
|------|---------|
| `app/actions.ts` | Server Actions for search, store, health |
| `app/api/health/route.ts` | Health check proxy |
| `app/api/search/route.ts` | Search proxy to Python backend |
| `app/api/store/route.ts` | Metadata store proxy |
| `app/api/anonymize/route.ts` | Anonymization proxy (routes by file type) |
| `app/api/ipfs/upload/route.ts` | IPFS upload proxy to JS backend |
| `app/api/preview/[type]/route.ts` | Dynamic preview proxy |
| `app/api/proxy/route.ts` | General backend proxy |
| `types/ethereum.d.ts` | `window.ethereum` TypeScript declarations |

---

## Key Changes

### 1. TypeScript

All `.js` files converted to `.ts`/`.tsx`. Every component has typed props:

```typescript
// Before (React)
function UploadData({ onBack, isWalletConnected, walletAddress, onWalletConnect }) { ... }

// After (Next.js)
interface UploadDataProps {
  onBack: () => void;
  isWalletConnected: boolean;
  walletAddress: string;
  onWalletConnect: () => void;
}

export default function UploadData({ onBack, isWalletConnected, walletAddress, onWalletConnect }: UploadDataProps) { ... }
```

### 2. Client Components

All interactive components use the `"use client"` directive since they rely on `useState`, `useEffect`, browser APIs (MetaMask, File API, etc.).

### 3. Routing

React conditional rendering replaced with Next.js file-based routing:

```
React:   {currentView === "search" && <SearchData />}
Next.js: app/search/page.tsx → imports <SearchData />
```

### 4. Environment Variables

```
React:     REACT_APP_ENCRYPTION_KEY
Next.js:   NEXT_PUBLIC_ENCRYPTION_KEY

React:     REACT_APP_JS_BACKEND_URL
Next.js:   NEXT_PUBLIC_JS_BACKEND_URL
```

### 5. Image Handling

`<img>` tags replaced with `next/image` `<Image>` component. Blob URLs use `unoptimized` prop.

### 6. Configuration

| Config | Change |
|--------|--------|
| `tsconfig.json` | Target `ES2018`, strict mode, `next` plugin |
| `tailwind.config.js` | Content paths include `.ts`/`.tsx` extensions |
| `next.config.js` | Image optimization (avif/webp), compression, vendor splitting |

---

## What Was Preserved

- Client-side AES-256 encryption (with streaming for large files)
- IPFS upload pipeline via JS backend
- Blockchain hash storage via Ethers.js v6
- MetaMask wallet connect/disconnect
- File anonymization (Excel, images, DICOM)
- Search with advanced filters
- Preview modal with 5% data sample
- Purchase & download flow

---

## Build Verification

```
npm run build

 Compiled successfully in ~5s
 Linting and checking validity of types
 Generating static pages (13/13)
 Collecting build traces
 Finalizing page optimization

4 static pages + 7 dynamic API routes
~279 KB shared JS bundle
```

---

## Rollback

The React prototype is untouched at `prototype/`. To rollback:

```bash
cd prototype
npm install && npm start
```
