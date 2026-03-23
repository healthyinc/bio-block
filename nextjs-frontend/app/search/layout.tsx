import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Search Documents - Bio-Block',
  description: 'Search, preview, and purchase medical records securely',
};

export default function SearchLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
