# Hypothesis Lab Demo

Interactive UI for the Hypothesis Tree decision engine and verifiable on-chain attestation.

## Getting Started

```bash
# install dependencies
npm install

# run local dev server (port 5174, proxies /demo to backend on 3003)
npm run dev

# production build
npm run build
```

## Features

- **Automated Profiler:** Inspect column distributions, cardinality, and data warnings.
- **Decision Tree Navigation:** Step-by-step hypothesis formulation with branch forking and side-by-side comparison.
- **Statistical Testing:** Dynamic test recommendation (t-test, ANOVA, Mann-Whitney U, Kruskal-Wallis, Chi-Square, correlations).
- **Interactive Visualizations:** Boxplots, density curves, forest plots, and statistical power simulator.
- **On-Chain Attestation:** EIP-712 signing via MetaMask, IPFS artifact publishing, and Sepolia registry logging.
