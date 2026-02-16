import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Upload Document - Bio-Block',
  description: 'Upload and encrypt medical records with blockchain verification',
};

export default function UploadLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
