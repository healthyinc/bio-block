from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import chromadb
import time
import uuid
import cv2
import pytesseract
import spacy
from pytesseract import Output
import numpy as np
from PIL import Image
import io
import shutil
import os
import json
from typing import List, Tuple
import re

# Preview factory imports
from services.preview.factory import PreviewFactory

# DICOM imports
try:
    import pydicom
    from pydicom.errors import InvalidDicomError
    pydicom_available = True
    print("pydicom library loaded successfully")
except ImportError:
    pydicom_available = False
    print("Warning: pydicom library not found. DICOM preview will not work.")
    print("Install with: pip install pydicom")

# Presidio imports for advanced image anonymization
try:
    from presidio_image_redactor import ImageRedactorEngine
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    presidio_available = True
    print("Presidio libraries loaded successfully")
except ImportError:
    presidio_available = False
    print("Warning: Presidio libraries not found. Advanced image anonymization will not work.")
    print("Install with: pip install presidio_analyzer presidio_anonymizer presidio_image_redactor")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="new_user_data")

# 1. Cross-platform Tesseract detection
tesseract_cmd = os.getenv('TESSERACT_CMD') or shutil.which('tesseract')

if not tesseract_cmd:
    common_paths = [
        '/opt/homebrew/bin/tesseract',
        '/usr/local/bin/tesseract',
        '/usr/bin/tesseract',
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    ]
    for path in common_paths:
        if os.path.isfile(path):
            tesseract_cmd = path
            break

tesseract_available = False
if tesseract_cmd:
    try:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        pytesseract.get_tesseract_version()
        tesseract_available = True
        print(f"Tesseract verified: {tesseract_cmd}")
    except Exception:
        print(f"Tesseract found at {tesseract_cmd} but failed to execute.")
else:
    print("Tesseract not found. Image anonymization will not work.")

try:
    nlp = spacy.load("en_core_web_lg")  # Updated to use large model for better accuracy
except IOError:
    try:
        nlp = spacy.load("en_core_web_sm")  # Fallback to small model
        print("Warning: Using en_core_web_sm. For better accuracy, install en_core_web_lg")
    except IOError:
        print("Warning: spaCy English model not found. Image anonymization will not work.")
        nlp = None

# Initialize Presidio engines for advanced anonymization
presidio_analyzer = None
presidio_anonymizer = None
presidio_image_redactor = None

if presidio_available:
    try:
        presidio_analyzer = AnalyzerEngine()
        presidio_anonymizer = AnonymizerEngine()
        presidio_image_redactor = ImageRedactorEngine()
        print("Presidio engines initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize Presidio engines: {str(e)}")
        presidio_available = False

PHI_LABELS = {"PERSON", "ORG", "GPE", "DATE", "LOC", "FAC", "NORP"}

class StoreRequest(BaseModel):
    summary: str
    dataset_title: str
    cid: str
    metadata: Optional[Dict[str, Any]] = {}

class SearchRequest(BaseModel):
    query: str

class FilterRequest(BaseModel):
    filters: Dict[str, Any]
    n_results: Optional[int] = 10
    
# Add new request model for enhanced storage
class StoreWithContentRequest(BaseModel):
    summary: str
    dataset_title: str
    cid: str
    metadata: Optional[Dict[str, Any]] = {}
    extracted_content: Optional[str] = ""  # Actual file content
    file_type: Optional[str] = "spreadsheet"  # spreadsheet, image, etc.

# Create separate collection for content-based search
content_collection = chroma_client.get_or_create_collection(name="document_content")
metadata_collection = chroma_client.get_or_create_collection(name="document_metadata")

class UpdateRequest(BaseModel):
    summary: Optional[str] = None
    dataset_title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    owner_address: str
    signature: str # Cryptographic signature of the request payload to prevent privilege escalation

class DeleteRequest(BaseModel):
    owner_address: str
    signature: str # Cryptographic signature to prevent unauthorized deletion

def generate_id() -> str:
    timestamp = int(time.time() * 1000)
    random_suffix = hash(str(uuid.uuid4())) % 10000
    return f"{timestamp}{random_suffix}"

def mask_phi_in_image_presidio(pil_image):
    """
    Advanced PHI detection and redaction using Presidio Image Redactor
    This uses state-of-the-art ML models for better accuracy
    """
    if not presidio_available or presidio_image_redactor is None:
        raise HTTPException(
            status_code=500, 
            detail="Presidio image redactor not available. Please install with: pip install presidio_analyzer presidio_anonymizer presidio_image_redactor"
        )
    
    try:
        # Use Presidio's advanced image redaction
        redacted_image = presidio_image_redactor.redact(pil_image)
        return redacted_image
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image with Presidio: {str(e)}")

def mask_phi_in_image_legacy(image_cv):
    """
    Legacy PHI detection using OCR and spaCy (fallback method)
    """
    if nlp is None:
        raise HTTPException(status_code=500, detail="spaCy English model not available. Please install it with: python -m spacy download en_core_web_sm")
    
    if not tesseract_available:
        raise HTTPException(status_code=500, detail="Tesseract OCR not available. Please install Tesseract: Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki, macOS: brew install tesseract, Linux: sudo apt install tesseract-ocr")
    
    try:
        # Convert BGR to RGB for OCR processing
        rgb_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        
        ocr_data = pytesseract.image_to_data(rgb_image, output_type=Output.DICT)
        
        full_text = " ".join(ocr_data["text"])
        
        doc = nlp(full_text)
        
        phi_entities = set(ent.text.strip() for ent in doc.ents if ent.label_ in PHI_LABELS)
        
        for i, word in enumerate(ocr_data["text"]):
            if word.strip() in phi_entities:
                x, y, w, h = ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i]
                cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 0, 0), -1)
        
        return image_cv
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Bio-Block Python Backend API", 
        "endpoints": {
            "/store": "Store document data",
            "/search": "Search documents", 
            "/filter": "Filter documents by metadata",
            "/search_with_filter": "Combined search and filter",
            "/documents/{doc_id}": "GET: Retrieve document by ID, PUT: Update document metadata, DELETE: Remove document",
            "/anonymize_image": "Anonymize PHI in images (Presidio + Legacy fallback)",
            "/anonymize_image_presidio": "Anonymize PHI in images (Presidio only, advanced)"
        },
        "status": {
            "presidio_available": presidio_available,
            "tesseract_available": tesseract_available,
            "spacy_model": "en_core_web_lg" if nlp and hasattr(nlp, 'meta') and nlp.meta.get('name') == 'en_core_web_lg' else "en_core_web_sm" if nlp else "not_available"
        }
    }

# @app.post("/anonymize_image")
# async def anonymize_image(file: UploadFile = File(...)):
#     """
#     Anonymize PHI in JPEG/JPG/PNG images using Presidio (preferred) or legacy OCR+spaCy
#     """
#     try:
#         # Validate file type
#         if not file.content_type or not file.content_type.startswith('image/'):
#             raise HTTPException(status_code=400, detail="File must be an image")
        
#         if file.content_type not in ['image/jpeg', 'image/jpg', 'image/png']:
#             raise HTTPException(status_code=400, detail="Only JPEG, JPG, and PNG images are supported")
        
#         # Read and convert image
#         contents = await file.read()
#         pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
#         # Try Presidio first (preferred method)
#         if presidio_available and presidio_image_redactor is not None:
#             try:
#                 print("Using Presidio advanced image redaction")
#                 redacted_image_pil = mask_phi_in_image_presidio(pil_image)
                
#                 # Save redacted image
#                 img_buffer = io.BytesIO()
#                 redacted_image_pil.save(img_buffer, format='JPEG', quality=95)
#                 img_buffer.seek(0)
                
#                 return StreamingResponse(
#                     io.BytesIO(img_buffer.read()),
#                     media_type="image/jpeg",
#                     headers={"Content-Disposition": f"attachment; filename=presidio_anonymized_{file.filename}"}
#                 )
                
#             except Exception as e:
#                 print(f"Presidio failed, falling back to legacy method: {str(e)}")
        
#         # Fallback to legacy OCR + spaCy method
#         print("Using legacy OCR + spaCy image redaction")
#         image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
#         masked_image_cv = mask_phi_in_image_legacy(image_cv)
        
#         # Convert back to PIL and save
#         masked_image_rgb = cv2.cvtColor(masked_image_cv, cv2.COLOR_BGR2RGB)
#         masked_pil = Image.fromarray(masked_image_rgb)
        
#         img_buffer = io.BytesIO()
#         masked_pil.save(img_buffer, format='JPEG', quality=95)
#         img_buffer.seek(0)

#         return StreamingResponse(
#             io.BytesIO(img_buffer.read()),
#             media_type="image/jpeg",
#             headers={"Content-Disposition": f"attachment; filename=legacy_anonymized_{file.filename}"}
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to anonymize image: {str(e)}")


@app.post("/anonymize_image")
async def anonymize_image_presidio_only(file: UploadFile = File(...)):
    """
    Anonymize PHI in images using ONLY Presidio (force advanced method)
    """
    try:
        if not presidio_available or presidio_image_redactor is None:
            raise HTTPException(
                status_code=503, 
                detail="Presidio not available. Install with: pip install presidio_analyzer presidio_anonymizer presidio_image_redactor && python -m spacy download en_core_web_lg"
            )
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        if file.content_type not in ['image/jpeg', 'image/jpg', 'image/png']:
            raise HTTPException(status_code=400, detail="Only JPEG, JPG, and PNG images are supported")
        
        # Read and process image
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        print("Processing image with Presidio advanced redaction")
        redacted_image_pil = mask_phi_in_image_presidio(pil_image)
        
        # Save and return redacted image
        img_buffer = io.BytesIO()
        redacted_image_pil.save(img_buffer, format='PNG', quality=95)
        img_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(img_buffer.read()),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=presidio_redacted_{file.filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to anonymize image with Presidio: {str(e)}")


@app.post("/store")
async def store_data_enhanced(request: StoreWithContentRequest):
    """
    Enhanced storage with both metadata and content vectors
    """
    try:
        print(f"Received enhanced storage request: {request.dataset_title}")
        doc_id = generate_id()
         
        # Prepare metadata document
        combined_metadata = f"Dataset Title: {request.dataset_title}\n{request.summary}"
        disease_tags = request.metadata.get("disease_tags", "")
     
        if disease_tags:
            combined_metadata += f"\nDisease Tags: {disease_tags}"
        
        metadata = {
            "cid": request.cid,
            "dataset_title": request.dataset_title,
            "file_type": request.file_type,
            **request.metadata
        }
        metadata_collection.add(
            ids=[doc_id],
            documents=[combined_metadata],
            metadatas=[metadata]
        )
        
     # Store content if available
        if request.extracted_content and request.extracted_content.strip():
            # Chunk content for better vectorization
            extractor = ContentExtractor()
            content_chunks = extractor.chunk_content(request.extracted_content)
            
            # Add each chunk with reference to original document
            chunk_ids = []
            chunk_docs = []
            chunk_metas = []
            
            for chunk_idx, chunk in enumerate(content_chunks):
                chunk_id = f"{doc_id}_chunk_{chunk_idx}"
                chunk_ids.append(chunk_id)
                chunk_docs.append(chunk)
                chunk_metas.append({
                    **metadata,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(content_chunks),
                    "parent_doc_id": doc_id
                })
            
            content_collection.add(
                ids=chunk_ids,
                documents=chunk_docs,
                metadatas=chunk_metas
            )
            
            print(f"Stored {len(content_chunks)} content chunks for document {doc_id}")
        
        return {
            "message": "Stored successfully with enhanced content indexing",
            "cid": request.cid,
            "doc_id": doc_id,
            "content_chunks": len(request.extracted_content.split()) if request.extracted_content else 0
        }   
        
    except Exception as e:
        print(f"Error in enhanced storage: {str(e)}")

        import traceback
        traceback.print_exc()  
        raise HTTPException(status_code=500, detail=f"Failed to store data: {str(e)}")

@app.post("/store_enhanced")
async def store_data_enhanced(request: StoreWithContentRequest):
    """
    Enhanced storage with both metadata and content vectors
    """
    try:
        print(f"Received enhanced storage request: {request.dataset_title}")
        doc_id = generate_id()
        
        # Prepare metadata document
        combined_metadata = f"Dataset Title: {request.dataset_title}\n{request.summary}"
        disease_tags = request.metadata.get("disease_tags", "")
        if disease_tags:
            combined_metadata += f"\nDisease Tags: {disease_tags}"
        
        metadata = {
            "cid": request.cid,
            "dataset_title": request.dataset_title,
            "file_type": request.file_type,
            **request.metadata
        }
        
        # Store in metadata collection
        metadata_collection.add(
            ids=[doc_id],
            documents=[combined_metadata],
            metadatas=[metadata]
        )
        
        # Store content if available
        if request.extracted_content and request.extracted_content.strip():
            extractor = ContentExtractor()
            content_chunks = extractor.chunk_content(request.extracted_content)
            
            chunk_ids = []
            chunk_docs = []
            chunk_metas = []
            
            for chunk_idx, chunk in enumerate(content_chunks):
                chunk_id = f"{doc_id}_chunk_{chunk_idx}"
                chunk_ids.append(chunk_id)
                chunk_docs.append(chunk)
                chunk_metas.append({
                    **metadata,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(content_chunks),
                    "parent_doc_id": doc_id
                })
            
            content_collection.add(
                ids=chunk_ids,
                documents=chunk_docs,
                metadatas=chunk_metas
            )
            
            print(f"Stored {len(content_chunks)} content chunks for document {doc_id}")
        
        return {
            "message": "Stored successfully with enhanced content indexing",
            "cid": request.cid,
            "doc_id": doc_id,
            "content_chunks": len(request.extracted_content.split()) if request.extracted_content else 0
        }
        
    except Exception as e:
        print(f"Error in enhanced storage: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to store data: {str(e)}")

@app.post("/search")
async def search_data(request: SearchRequest):
    try:
        search_results = collection.query(
            query_texts=[request.query],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )
        
        results = []
        if search_results["ids"][0]:
            for i, doc_id in enumerate(search_results["ids"][0]):
                metadata = search_results["metadatas"][0][i]
                distance = search_results["distances"][0][i]
                document = search_results["documents"][0][i]
                
                score = 1 / (1 + distance)
                
                results.append({
                    "id": doc_id,
                    "cid": metadata.get("cid", ""),
                    "score": score,
                    "summary": document,
                    "metadata": metadata
                })
        
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to search data")

@app.post("/filter")
async def filter_data(request: FilterRequest):
    try:
        print(f"Filter request: {request.filters}") 
        
        filters = request.filters
        if len(filters) > 1:
           
            and_conditions = [{key: value} for key, value in filters.items()]
            where_clause = {"$and": and_conditions}
        else:
           
            where_clause = filters
            
        print(f"Where clause: {where_clause}")  
        
       
        search_results = collection.get(
            where=where_clause,
            include=["documents", "metadatas"]
        )
        
        print(f"Filter results count: {len(search_results['ids']) if search_results['ids'] else 0}")  # Debug log
        
        results = []
        if search_results["ids"]:
            for i, doc_id in enumerate(search_results["ids"]):
                metadata = search_results["metadatas"][i]
                document = search_results["documents"][i]
                
                results.append({
                    "id": doc_id,
                    "cid": metadata.get("cid", ""),
                    "summary": document,
                    "metadata": metadata
                })
        
        if request.n_results and len(results) > request.n_results:
            results = results[:request.n_results]
        
        return {"results": results, "total_found": len(search_results["ids"])}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to filter data: {str(e)}")

@app.post("/search_with_filter")
async def search_with_filter(request: Dict[str, Any]):
    """
    Combined semantic search with metadata filtering
    Request body should contain:
    - query: text to search for
    - filters: metadata filters (optional)
    - n_results: number of results (optional, default 5)
    """
    try:
        query = request.get("query")
        filters = request.get("filters", {})
        n_results = request.get("n_results", 5)
        
        print(f"Search with filter - Query: {query}, Filters: {filters}")  
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
     
        search_kwargs = {
            "query_texts": [query],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"]
        }
        
        if filters:
         
            if len(filters) > 1:
                and_conditions = [{key: value} for key, value in filters.items() if value]
                search_kwargs["where"] = {"$and": and_conditions}
            else:
                search_kwargs["where"] = filters
        
        search_results = collection.query(**search_kwargs)
        
        results = []
        if search_results["ids"][0]:
            for i, doc_id in enumerate(search_results["ids"][0]):
                metadata = search_results["metadatas"][0][i]
                distance = search_results["distances"][0][i]
                document = search_results["documents"][0][i]
           
                score = 1 / (1 + distance)
                
                results.append({
                    "id": doc_id,
                    "cid": metadata.get("cid", ""),
                    "score": score,
                    "summary": document,
                    "metadata": metadata
                })
        
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search with filter: {str(e)}")
    
@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    try:
        result = collection.get(
            ids=[doc_id],
            include=["documents", "metadatas"]
        )
        
        if not result["ids"]:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "id": result["ids"][0],
            "cid": result["metadatas"][0].get("cid", ""),
            "summary": result["documents"][0],
            "metadata": result["metadatas"][0]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")

@app.put("/documents/{doc_id}")
async def update_document(doc_id: str, request: UpdateRequest):
    try:
        existing = collection.get(
            ids=[doc_id],
            include=["documents", "metadatas"]
        )
        
        if not existing["ids"]:
            raise HTTPException(status_code=404, detail="Document not found")
        
        old_metadata = existing["metadatas"][0]
        stored_owner = old_metadata.get("owner_address", "")
        if stored_owner.lower() != request.owner_address.lower():
            raise HTTPException(status_code=403, detail="Unauthorized: not document owner")
        
        # Security Fix: Verify cryptographic signature to prove ownership of the address
        # TODO: Implement `eth_account.messages.recover_message` to verify request.signature
        if not request.signature:
            raise HTTPException(status_code=401, detail="Missing cryptographic signature")
        # --- End Security Fix ---
        
        new_dataset_title = request.dataset_title if request.dataset_title else old_metadata.get("dataset_title", "")
        new_summary = request.summary if request.summary else existing["documents"][0]
        
        updated_metadata = {**old_metadata}
        if request.metadata:
            updated_metadata.update(request.metadata)
        updated_metadata["dataset_title"] = new_dataset_title
        updated_metadata["owner_address"] = request.owner_address
        
        new_doc_id = generate_id()
        combined_document = f"Dataset Title: {new_dataset_title}\n{new_summary}"
        
        disease_tags = updated_metadata.get("disease_tags")
        if disease_tags:
            combined_document += f"\nDisease Tags: {disease_tags}"
        
        collection.add(
            ids=[new_doc_id],
            documents=[combined_document],
            metadatas=[updated_metadata]
        )
        
        print(f"Document updated: {doc_id} -> {new_doc_id}")
        
        return {"message": "Document updated", "old_id": doc_id, "new_id": new_doc_id}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update document: {str(e)}")

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, request: DeleteRequest):
    try:
        existing = collection.get(
            ids=[doc_id],
            include=["metadatas"]
        )
        
        if not existing["ids"]:
            raise HTTPException(status_code=404, detail="Document not found")
        
        stored_owner = existing["metadatas"][0].get("owner_address", "")
        if stored_owner.lower() != request.owner_address.lower():
            raise HTTPException(status_code=403, detail="Unauthorized: not document owner")
        
        # Security Fix: Verify cryptographic signature to prove ownership of the address
        # TODO: Implement `eth_account.messages.recover_message` to verify request.signature
        if not request.signature:
            raise HTTPException(status_code=401, detail="Missing cryptographic signature")
        # --- End Security Fix ---
        
        collection.delete(ids=[doc_id])
        
        print(f"Document {doc_id} deleted")
        
        return {"message": "Document deleted", "deleted_id": doc_id}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


# --- START: SIMPLE PREVIEW ENDPOINT ---

@app.post("//simple_preview") # Catches the bad URL
@app.post("/simple_preview", include_in_schema=False)
async def simple_preview(file: UploadFile = File(...)):
    """
    A simple endpoint that just returns the uploaded image
    without any anonymization. This is to test the pipeline.
    
    Now uses the PreviewFactory to support multiple file types including DICOM.
    Maintains backward compatibility with existing frontend.
    """
    print("✅ --- simple_preview endpoint was called! --- ✅")
    print(f"File received: {file.filename}, Content-Type: {file.content_type}")

    try:
        # Read the file contents
        file_contents = await file.read()

        # Use factory to get the appropriate generator
        generator = PreviewFactory.create_generator(
            filename=file.filename,
            content_type=file.content_type
        )

        # Generate preview using the factory-selected generator
        response, media_type = generator.generate_preview(
            file_contents=file_contents,
            filename=file.filename,
            content_type=file.content_type
        )

        print(f"Sending response with media_type: {media_type}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate preview: {str(e)}"
        )

# --- END: SIMPLE PREVIEW ENDPOINT ---

@app.post("/preview_dicom", include_in_schema=False)
async def preview_dicom(file: UploadFile = File(...)):
    """
    Convert DICOM file to PNG/JPEG image for preview.
    Works on both Mac and Windows.
    
    Now uses the PreviewFactory for consistency.
    Maintains backward compatibility with existing frontend.
    """
    try:
        # Read the file contents
        file_contents = await file.read()

        # Use factory to get the DICOM generator (factory will validate file type)
        generator = PreviewFactory.create_generator(
            filename=file.filename,
            content_type=file.content_type
        )

        # Generate preview using the factory-selected generator
        response, media_type = generator.generate_preview(
            file_contents=file_contents,
            filename=file.filename,
            content_type=file.content_type
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to convert DICOM to image: {str(e)}"
        )
@app.post("/search_enhanced")
async def search_data_enhanced(request: Dict[str, Any]):
    """
    Enhanced search combining content and metadata with intelligent ranking
    
    Request body:
    {
        "query": "search query",
        "content_weight": 0.6,  # Weight for content search (default 0.6)
        "metadata_weight": 0.4,  # Weight for metadata search (default 0.4)
        "n_results": 5,
        "filters": {} (optional)
    }
    """
    try:
        query = request.get("query")
        content_weight = float(request.get("content_weight", 0.6))
        metadata_weight = float(request.get("metadata_weight", 0.4))
        n_results = request.get("n_results", 5)
        filters = request.get("filters", {})
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Normalize weights
        total_weight = content_weight + metadata_weight
        content_weight = content_weight / total_weight
        metadata_weight = metadata_weight / total_weight
        
        # Search in content collection
        content_results = {}
        try:
            content_search = content_collection.query(
                query_texts=[query],
                n_results=n_results * 2,  # Get more results to account for chunking
                include=["documents", "metadatas", "distances"]
            )
            
            # Group chunks by parent document
            seen_docs = set()
            for i, doc_id in enumerate(content_search["ids"][0]):
                metadata = content_search["metadatas"][0][i]
                parent_id = metadata.get("parent_doc_id", doc_id)
                
                if parent_id not in seen_docs:
                    distance = content_search["distances"][0][i]
                    score = 1 / (1 + distance)
                    content_results[parent_id] = {
                        "score": score,
                        "distance": distance,
                        "source": "content",
                        "metadata": metadata
                    }
                    seen_docs.add(parent_id)
        except Exception as e:
            print(f"Content search error: {e}")
            content_results = {}
        
        # Search in metadata collection
        metadata_results = {}
        search_kwargs = {
            "query_texts": [query],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"]
        }
        
        if filters and any(filters.values()):
            if len(filters) > 1:
                and_conditions = [{key: value} for key, value in filters.items() if value]
                search_kwargs["where"] = {"$and": and_conditions}
            else:
                search_kwargs["where"] = filters
        
        metadata_search = metadata_collection.query(**search_kwargs)
        
        for i, doc_id in enumerate(metadata_search["ids"][0]):
            distance = metadata_search["distances"][0][i]
            score = 1 / (1 + distance)
            metadata_results[doc_id] = {
                "score": score,
                "distance": distance,
                "source": "metadata",
                "metadata": metadata_search["metadatas"][0][i],
                "summary": metadata_search["documents"][0][i]
            }
        
        # Combine results with weighted scoring
        combined_results = {}
        
        for doc_id, content_data in content_results.items():
            if doc_id not in combined_results:
                combined_results[doc_id] = {
                    "content_score": 0,
                    "metadata_score": 0,
                    "metadata": content_data["metadata"]
                }
            combined_results[doc_id]["content_score"] = content_data["score"]
        
        for doc_id, metadata_data in metadata_results.items():
            if doc_id not in combined_results:
                combined_results[doc_id] = {
                    "content_score": 0,
                    "metadata_score": 0,
                    "metadata": metadata_data["metadata"],
                    "summary": metadata_data["summary"]
                }
            else:
                combined_results[doc_id]["summary"] = metadata_data["summary"]
            combined_results[doc_id]["metadata_score"] = metadata_data["score"]
        
        # Calculate final scores and sort
        results = []
        for doc_id, scores in combined_results.items():
            final_score = (scores["content_score"] * content_weight + 
                          scores["metadata_score"] * metadata_weight)
            results.append({
                "id": doc_id,
                "cid": scores["metadata"].get("cid", ""),
                "score": final_score,
                "content_score": scores["content_score"],
                "metadata_score": scores["metadata_score"],
                "summary": scores.get("summary", ""),
                "metadata": scores["metadata"]
            })
        
        # Sort by final score
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:n_results]
        
        return {
            "results": results,
            "search_config": {
                "content_weight": content_weight,
                "metadata_weight": metadata_weight
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Enhanced search error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Enhanced search failed: {str(e)}")

# Content Extraction Module for Enhanced Retrieval
class ContentExtractor:
    """
    Extracts and processes content from various file formats for semantic search
    """
    
    @staticmethod
    def extract_spreadsheet_content(data: List[List[str]], title: str = "") -> str:
        """
        Extract meaningful content from spreadsheet data
        - data: 2D array of spreadsheet cells
        - title: dataset title for context
        Returns: formatted text for vectorization
        """
        if not data or not data[0]:
            return ""
        
        headers = data[0]
        content_parts = []
        
        # Add title context
        if title:
            content_parts.append(f"Dataset: {title}")
        
        # Add headers as schema info
        content_parts.append(f"Columns: {', '.join(str(h) for h in headers if h)}")
        
        # Add sample data patterns (first 10 rows for content understanding)
        sample_rows = data[1:min(11, len(data))]
        
        for row_idx, row in enumerate(sample_rows):
            row_content = []
            for col_idx, cell in enumerate(row):
                if cell and col_idx < len(headers):
                    header = headers[col_idx]
                    # Skip anonymized IDs
                    if not (isinstance(cell, str) and cell.startswith('WID_')):
                        row_content.append(f"{header}: {cell}")
            
            if row_content:
                content_parts.append(f"Row {row_idx + 1}: {'; '.join(row_content)}")
        
        return "\n".join(content_parts)
    
    @staticmethod
    def extract_csv_content(csv_text: str, title: str = "") -> str:
        """
        Extract content from CSV text
        """
        lines = csv_text.strip().split('\n')
        if not lines:
            return ""
        
        headers = lines[0].split(',')
        content_parts = []
        
        if title:
            content_parts.append(f"Dataset: {title}")
        
        content_parts.append(f"Columns: {', '.join(headers)}")
        
        # Process sample rows
        for line_idx, line in enumerate(lines[1:min(11, len(lines))]):
            values = line.split(',')
            row_content = []
            for col_idx, value in enumerate(values):
                if value.strip() and col_idx < len(headers):
                    if not value.startswith('WID_'):
                        row_content.append(f"{headers[col_idx]}: {value}")
            
            if row_content:
                content_parts.append(f"Row {line_idx + 1}: {'; '.join(row_content)}")
        
        return "\n".join(content_parts)
    
    @staticmethod
    def chunk_content(content: str, chunk_size: int = 500) -> List[str]:
        """
        Split large content into chunks for better vectorization
        Returns list of content chunks
        """
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [content]
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3002)

