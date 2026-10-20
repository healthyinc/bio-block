with open("nextjs-frontend/components/UploadData.tsx", "r") as f:
    content = f.read()

content = content.replace("import { storeDocumentHash } from '../lib/contractService';", "import { storeDocumentHash } from '../lib/contractService';\nimport { generateDocumentKey } from '../lib/encryptionUtils';")

content = content.replace("const uploadToIPFSViaBackend = async (\n    encryptedData: Blob | Uint8Array,\n    fileName: string\n  ): Promise<IPFSUploadResult> => {", "const uploadToIPFSViaBackend = async (\n    encryptedData: Blob | Uint8Array,\n    fileName: string,\n    documentKey: string\n  ): Promise<IPFSUploadResult> => {")

content = content.replace("formData.append('fileName', fileName);", "formData.append('fileName', fileName);\n    formData.append('documentKey', documentKey);")

content = content.replace("""      try {
        const streamer = new StreamingEncryption();
        
        const shouldUseStreaming = streamer.shouldUseStreaming(fileToUpload.size);""", """      try {
        const documentKey = generateDocumentKey();
        const streamer = new StreamingEncryption(documentKey);
        
        const shouldUseStreaming = streamer.shouldUseStreaming(fileToUpload.size);""")

content = content.replace("""        } else {
          const fileBuffer = await fileToUpload.arrayBuffer();
          encryptedFile = encryptFile(new Uint8Array(fileBuffer));
        }""", """        } else {
          const fileBuffer = await fileToUpload.arrayBuffer();
          encryptedFile = encryptFile(new Uint8Array(fileBuffer), documentKey);
        }""")

content = content.replace("""        const result = await uploadToIPFSViaBackend(
          encryptedFile,
          fileToUpload.name
        );""", """        const result = await uploadToIPFSViaBackend(
          encryptedFile,
          fileToUpload.name,
          documentKey
        );""")

content = content.replace("""          try {
            const previewStreamer = new StreamingEncryption();
            const shouldUseStreamingForPreview =
              previewStreamer.shouldUseStreaming(previewFile.size);""", """          try {
            const previewKey = generateDocumentKey();
            const previewStreamer = new StreamingEncryption(previewKey);
            const shouldUseStreamingForPreview =
              previewStreamer.shouldUseStreaming(previewFile.size);""")

content = content.replace("""            } else {
              const previewBuffer = await previewFile.arrayBuffer();
              encryptedPreview = encryptFile(new Uint8Array(previewBuffer));
            }""", """            } else {
              const previewBuffer = await previewFile.arrayBuffer();
              encryptedPreview = encryptFile(new Uint8Array(previewBuffer), previewKey);
            }""")

content = content.replace("""            const previewResult = await uploadToIPFSViaBackend(
              encryptedPreview,
              previewFile.name
            );""", """            const previewResult = await uploadToIPFSViaBackend(
              encryptedPreview,
              previewFile.name,
              previewKey
            );""")

with open("nextjs-frontend/components/UploadData.tsx", "w") as f:
    f.write(content)
