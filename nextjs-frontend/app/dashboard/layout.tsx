import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Dashboard - Bio-Block',
  description: 'Manage your medical records, view earnings, and withdraw funds',
};

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
