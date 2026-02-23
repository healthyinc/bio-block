import { Shield, Database, Zap } from "lucide-react";

export default function HeroSection() {
  return (
    <div className="text-center mb-12">
      <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl mb-6 shadow-lg">
        <Shield size={32} className="text-white" />
      </div>
      <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent mb-4">
        Bio-Block
      </h1>
      <p className="text-xl text-gray-600 mb-2 max-w-2xl mx-auto leading-relaxed">
        Secure, decentralized document management powered by blockchain technology
      </p>
      <p className="text-gray-500 max-w-xl mx-auto mb-12">
        Upload, search, and manage your medical documents with enterprise-grade security and privacy
      </p>

      {/* Features Grid */}
      <div className="grid md:grid-cols-3 gap-6">
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-lg hover:shadow-xl transition-all duration-200 border border-gray-100">
          <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center mb-4">
            <Database size={24} className="text-blue-600" />
          </div>
          <h3 className="font-semibold text-gray-800 mb-2">IPFS Storage</h3>
          <p className="text-sm text-gray-600">Decentralized storage ensures your documents are always accessible and secure</p>
        </div>
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-lg hover:shadow-xl transition-all duration-200 border border-gray-100">
          <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center mb-4">
            <Shield size={24} className="text-green-600" />
          </div>
          <h3 className="font-semibold text-gray-800 mb-2">Blockchain Verified</h3>
          <p className="text-sm text-gray-600">Ethereum blockchain ensures document integrity and ownership verification</p>
        </div>
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-lg hover:shadow-xl transition-all duration-200 border border-gray-100">
          <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center mb-4">
            <Zap size={24} className="text-purple-600" />
          </div>
          <h3 className="font-semibold text-gray-800 mb-2">Smart Search</h3>
          <p className="text-sm text-gray-600">Advanced filtering and semantic search to find exactly what you need</p>
        </div>
      </div>
    </div>
  );
}
