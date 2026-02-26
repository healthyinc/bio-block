"use client";

import { useRouter } from 'next/navigation';
import SearchData from '../../components/SearchData';

export default function SearchPage() {
  const router = useRouter();

  const handleBack = () => {
    router.push('/');
  };

  return <SearchData onBack={handleBack} />;
}
