import "./globals.css";
import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Bio-Block - Decentralized Medical Records",
  description:
    "Secure, encrypted, and decentralized medical record storage using blockchain and IPFS",
  keywords: ["blockchain", "medical records", "IPFS", "encryption", "healthcare", "PHI", "decentralized"],
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

interface RootLayoutProps {
  children: React.ReactNode;
}

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="en">
      <body className="m-0 p-0">{children}</body>
    </html>
  );
}
