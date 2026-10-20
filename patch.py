with open("nextjs-frontend/components/SearchData.tsx", "r") as f:
    content = f.read()

# Replace smartDecrypt
content = content.replace("""  // Smart decryption helper
  const smartDecrypt = async (
    encryptedData: string | Uint8Array,
    progressCallback?: (progress: number) => void
  ): Promise<string | Uint8Array> => {
    try {
      const dataString =
        typeof encryptedData === 'string'
          ? encryptedData
          : encryptedData instanceof Uint8Array
          ? Buffer.from(encryptedData).toString('utf8')
          : String(encryptedData);

      if (
        dataString.includes('|METADATA_SEPARATOR|') &&
        dataString.includes('|CHUNK_SEPARATOR|')
      ) {
        const streamer = new StreamingEncryption();
        return await streamer.decryptFileStream(dataString, progressCallback || null);
      } else {
        return decryptFile(encryptedData);
      }
    } catch (error) {
      return decryptFile(encryptedData);
    }
  };""", """  const getDocumentKey = async (ipfsHash: string): Promise<string> => {
    // @ts-ignore - ethereum is injected by MetaMask
    if (!window.ethereum) throw new Error("MetaMask not connected");
    // @ts-ignore
    const accounts = await window.ethereum.request({ method: 'eth_accounts' });
    if (accounts.length === 0) throw new Error("No connected account found");
    const buyerAddress = accounts[0];

    const backendUrl = process.env.NEXT_PUBLIC_JS_BACKEND_URL || 'http://localhost:3001';
    const response = await fetch(`${backendUrl}/api/ipfs/key/${ipfsHash}?buyerAddress=${buyerAddress}`);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `Failed to get decryption key (${response.status})`);
    }

    const data = await response.json();
    return data.documentKey;
  };

  // Smart decryption helper
  const smartDecrypt = async (
    encryptedData: string | Uint8Array,
    documentKey: string,
    progressCallback?: (progress: number) => void
  ): Promise<string | Uint8Array> => {
    try {
      const dataString =
        typeof encryptedData === 'string'
          ? encryptedData
          : encryptedData instanceof Uint8Array
          ? Buffer.from(encryptedData).toString('utf8')
          : String(encryptedData);

      if (
        dataString.includes('|METADATA_SEPARATOR|') &&
        dataString.includes('|CHUNK_SEPARATOR|')
      ) {
        const streamer = new StreamingEncryption(documentKey);
        return await streamer.decryptFileStream(dataString, progressCallback || null);
      } else {
        return decryptFile(encryptedData, documentKey);
      }
    } catch (error) {
      return decryptFile(encryptedData, documentKey);
    }
  };""")

# Now replace the calls
content = content.replace("""      const response = await fetch(`https://gateway.pinata.cloud/ipfs/${cid}`);
      const encryptedData = await response.text();
      
      const decryptedData = await smartDecrypt(
        encryptedData,
        () => {}
      );""", """      const response = await fetch(`https://gateway.pinata.cloud/ipfs/${cid}`);
      const encryptedData = await response.text();
      
      const documentKey = await getDocumentKey(cid);
      
      const decryptedData = await smartDecrypt(
        encryptedData,
        documentKey,
        () => {}
      );""")

content = content.replace("""      const response = await fetch(`https://gateway.pinata.cloud/ipfs/${previewHash}`);
      const encryptedData = await response.text();
      
      const decryptedData = await smartDecrypt(
        encryptedData,
        () => {}
      );""", """      const response = await fetch(`https://gateway.pinata.cloud/ipfs/${previewHash}`);
      const encryptedData = await response.text();
      
      const previewKey = await getDocumentKey(previewHash);
      
      const decryptedData = await smartDecrypt(
        encryptedData,
        previewKey,
        () => {}
      );""")

with open("nextjs-frontend/components/SearchData.tsx", "w") as f:
    f.write(content)
