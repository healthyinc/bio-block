import "./globals.css";

export const metadata = {
  title: "Bio-Block - Decentralized Medical Records",
  description:
    "Secure, encrypted, and decentralized medical record storage using blockchain and IPFS",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
