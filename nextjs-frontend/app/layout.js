import "./globals.css";

export const metadata = {
  title: "Bio-Block - Decentralized Medical Records",
  description:
    "Secure, encrypted, and decentralized medical record storage using blockchain and IPFS",
  keywords: [
    "blockchain",
    "medical records",
    "IPFS",
    "encryption",
    "healthcare",
    "PHI",
    "decentralized",
  ],
  authors: [{ name: "Bio-Block Team" }],
  openGraph: {
    title: "Bio-Block - Secure Medical Records on Blockchain",
    description: "Store, manage, and share your medical records with complete privacy and security",
    type: "website",
    locale: "en_US",
  },
  twitter: {
    card: "summary_large_image",
    title: "Bio-Block - Decentralized Medical Records",
    description: "Secure medical record storage using blockchain and IPFS",
  },
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
