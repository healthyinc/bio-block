import { Shield, Database, Globe } from "lucide-react";
import FeatureCard from "./FeatureCard";

export default function HeroSection() {
  return (
    <div className="max-w-4xl mx-auto text-center">
      <h2 className="text-4xl font-bold text-gray-800 mb-6">
        Secure Medical Records on Blockchain
      </h2>
      <p className="text-xl text-gray-600 mb-8">
        Store, manage, and share your medical records with complete privacy and security
      </p>

      <div className="grid md:grid-cols-3 gap-8 mt-12">
        <FeatureCard
          icon={Shield}
          title="Client-Side Encryption"
          description="Your files are encrypted in your browser before upload"
        />
        <FeatureCard
          icon={Database}
          title="IPFS Storage"
          description="Decentralized storage ensures your data is always accessible"
        />
        <FeatureCard
          icon={Globe}
          title="Blockchain Security"
          description="Immutable records stored on Ethereum blockchain"
        />
      </div>

      <div className="mt-12 bg-yellow-50 border-2 border-yellow-200 rounded-xl p-6">
        <h3 className="text-xl font-semibold text-yellow-800 mb-2">
          Connect your wallet to get started
        </h3>
        <p className="text-yellow-700">
          Make sure you have MetaMask or another Web3 wallet installed
        </p>
      </div>
    </div>
  );
}
