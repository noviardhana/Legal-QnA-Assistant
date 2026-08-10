"""
Legal Document Index Builder

This script builds a FAISS vector index from legal PDF documents (Indonesian law documents).
Run this ONCE locally (Colab/PC with access to the PDF source) to generate indexes 
used by app.py on Streamlit Cloud.

Outputs:
  - index/faiss_index/      : FAISS vector store for semantic search
  - index/child_docs.pkl    : List of small document chunks (for BM25 + parent_id references)
  - index/parent_docs.pkl   : Dict mapping parent_id -> large document chunks (for LLM context)

After running, commit the `index/` folder to GitHub so Streamlit Cloud
doesn't need to reprocess PDFs on every startup.

The script:
  1. Checks if legal_docs/ folder exists; downloads PDFs from Google Drive if missing
  2. Loads PDF files and extracts metadata (law number, year)
  3. Creates parent chunks (large context) and child chunks (small, for retrieval)
  4. Generates embeddings using multilingual HuggingFace model
  5. Builds and saves FAISS index
  6. Serializes documents for runtime retrieval
"""

import os
import pickle
import re
import uuid
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================================================
# Configuration
# =========================================================
LEGAL_DOCS_DIR = "legal_docs"
INDEX_DIR = "index"
FAISS_DIR = os.path.join(INDEX_DIR, "faiss_index")
GOOGLE_DRIVE_FOLDER_ID = "1LHZ1IncPmmUN5kytFu3i7MoaafFrKDql"

# Text splitting parameters
PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 50

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# =========================================================
# Utility Functions
# =========================================================
def download_legal_docs_if_missing():
    """
    Check if legal_docs folder and PDFs exist.
    If missing, download from Google Drive using gdown.
    
    Returns:
        bool: True if PDFs are available, False if download failed.
    """
    os.makedirs(LEGAL_DOCS_DIR, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(LEGAL_DOCS_DIR) if f.lower().endswith(".pdf")]
    
    if len(pdf_files) == 4:
        print(f"✓ Found 4 PDFs in {LEGAL_DOCS_DIR}/")
        return True
    
    print(f"⚠ Only {len(pdf_files)} PDFs found. Attempting to download from Google Drive...")
    
    try:
        import gdown
    except ImportError:
        print("❌ gdown not installed. Install with: pip install gdown")
        return False
    
    try:
        print(f"Downloading from Google Drive folder ID: {GOOGLE_DRIVE_FOLDER_ID}")
        gdown.download_folder(
            f"https://drive.google.com/drive/folders/{GOOGLE_DRIVE_FOLDER_ID}",
            output=LEGAL_DOCS_DIR,
            quiet=False,
            use_cookies=False
        )
        
        # Re-check after download
        pdf_files = [f for f in os.listdir(LEGAL_DOCS_DIR) if f.lower().endswith(".pdf")]
        if len(pdf_files) == 4:
            print(f"✓ Successfully downloaded 4 PDFs to {LEGAL_DOCS_DIR}/")
            return True
        else:
            print(f"❌ Download completed but only {len(pdf_files)} PDFs found.")
            return False
    
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def extract_law_metadata(text):
    """
    Extract Indonesian law (Undang-Undang) number and year from document text.
    
    Searches for the pattern "UNDANG-UNDANG NOMOR X TAHUN YYYY" in the document.
    This pattern is standard in Indonesian legal documents.
    
    Args:
        text (str): Document text to search for law metadata (typically first 3000 chars).
    
    Returns:
        dict: Contains 'law_number' and 'law_year' keys.
              Defaults to {"law_number": "unknown", "law_year": "unknown"} if pattern not found.
    
    Example:
        >>> text = "UNDANG-UNDANG NOMOR 13 TAHUN 2003"
        >>> extract_law_metadata(text)
        {'law_number': '13', 'law_year': '2003'}
    """
    match = re.search(
        r"UNDANG[- ]?UNDANG\s+(?:REPUBLIK INDONESIA\s+)?NOMOR\s+(\d+)\s+TAHUN\s+(\d{4})",
        text.upper(),
    )
    if match:
        return {"law_number": match.group(1), "law_year": match.group(2)}
    return {"law_number": "unknown", "law_year": "unknown"}


def load_and_chunk_pdfs(parent_splitter, child_splitter):
    """
    Load PDF files and create a two-level hierarchy of document chunks.
    
    Creates two types of chunks:
    - Parent chunks: Large context windows (2000 chars) for LLM context
    - Child chunks: Small retrieval units (300 chars) for search and ranking
    
    Args:
        parent_splitter (RecursiveCharacterTextSplitter): Splitter for large chunks
        child_splitter (RecursiveCharacterTextSplitter): Splitter for small chunks
    
    Returns:
        tuple: (parent_docs dict, child_docs list)
               - parent_docs: {parent_id (str) -> Document}
               - child_docs: [Document, Document, ...]
    """
    parent_docs = {}
    child_docs = []
    
    pdf_files = sorted([f for f in os.listdir(LEGAL_DOCS_DIR) if f.lower().endswith(".pdf")])
    
    for fname in pdf_files:
        print(f"\nProcessing: {fname}")
        loader = PyPDFLoader(os.path.join(LEGAL_DOCS_DIR, fname))
        pages = loader.load()
        full_text = "\n".join(p.page_content for p in pages)
        law_meta = extract_law_metadata(full_text[:3000])
        
        print(f"  - Law Number: {law_meta.get('law_number')}/{law_meta.get('law_year')}")
        
        # Create parent chunks (large context)
        parents = parent_splitter.create_documents([full_text])
        print(f"  - Created {len(parents)} parent chunks")
        
        for p_idx, parent in enumerate(parents):
            parent_id = str(uuid.uuid4())
            parent.metadata.update({
                "source": fname,
                "parent_idx": p_idx,
                **law_meta
            })
            parent_docs[parent_id] = parent
            
            # Create child chunks (small, searchable)
            children = child_splitter.create_documents([parent.page_content])
            for c_idx, child in enumerate(children):
                child.metadata.update({
                    "source": fname,
                    "parent_id": parent_id,
                    "parent_idx": p_idx,
                    "child_idx": c_idx,
                    **law_meta,
                })
                child_docs.append(child)
    
    return parent_docs, child_docs


def build_faiss_index(child_docs):
    """
    Generate embeddings for child documents and build FAISS vector index.
    
    Uses multilingual sentence transformers to generate dense vector representations
    of document chunks. These embeddings enable semantic search in retrieval.
    
    Args:
        child_docs (list): List of Document objects to embed and index.
    
    Returns:
        FAISS: Initialized and saved FAISS vector store.
    """
    print(f"Using embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(child_docs, embeddings)
    vectorstore.save_local(FAISS_DIR)
    print(f"✓ FAISS index saved to {FAISS_DIR}/")
    return vectorstore


def save_documents(parent_docs, child_docs):
    """
    Serialize and pickle document collections for runtime use.
    
    Saves:
    - parent_docs.pkl: Large context documents (for LLM input)
    - child_docs.pkl: Small retrieval documents (for BM25 + semantic search)
    
    Args:
        parent_docs (dict): Mapping of parent_id -> Document
        child_docs (list): List of child Document objects
    """
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    with open(os.path.join(INDEX_DIR, "child_docs.pkl"), "wb") as f:
        pickle.dump(child_docs, f)
    print(f"✓ Saved child_docs.pkl ({len(child_docs)} documents)")
    
    with open(os.path.join(INDEX_DIR, "parent_docs.pkl"), "wb") as f:
        pickle.dump(parent_docs, f)
    print(f"✓ Saved parent_docs.pkl ({len(parent_docs)} documents)")


# =========================================================
# Main Indexing Pipeline
# =========================================================
def main():
    """
    Main orchestration function for the indexing pipeline.
    
    Steps:
    1. Check and download legal documents from Google Drive if missing
    2. Initialize text splitters
    3. Load PDFs and create document chunks
    4. Generate embeddings and build FAISS index
    5. Serialize and save all documents
    """
    # Step 1: Check and download PDFs if needed
    print("=" * 60)
    print("STEP 1: Checking legal documents...")
    print("=" * 60)
    if not download_legal_docs_if_missing():
        print("❌ Unable to proceed without legal documents.")
        return
    
    # Step 2: Prepare directories and text splitters
    print("\n" + "=" * 60)
    print("STEP 2: Initializing text splitters...")
    print("=" * 60)
    
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP
    )
    print("✓ Text splitters ready")
    
    # Step 3: Load PDFs and create document hierarchy
    print("\n" + "=" * 60)
    print("STEP 3: Loading PDFs and creating document chunks...")
    print("=" * 60)
    parent_docs, child_docs = load_and_chunk_pdfs(parent_splitter, child_splitter)
    print(f"\n✓ Total parent chunks: {len(parent_docs)}")
    print(f"✓ Total child chunks: {len(child_docs)}")
    
    # Step 4: Generate embeddings and build FAISS index
    print("\n" + "=" * 60)
    print("STEP 4: Building embeddings and FAISS index...")
    print("=" * 60)
    build_faiss_index(child_docs)
    
    # Step 5: Serialize and save pickled documents
    print("\n" + "=" * 60)
    print("STEP 5: Saving serialized documents...")
    print("=" * 60)
    save_documents(parent_docs, child_docs)
    
    # Completion message
    print("\n" + "=" * 60)
    print("✓ INDEXING COMPLETE")
    print("=" * 60)
    print(f"Index saved to: ./{INDEX_DIR}/")
    print("\nNext steps:")
    print("  1. Commit the 'index/' folder to your GitHub repository")
    print("  2. Streamlit Cloud will use this index automatically")
    print("  3. No re-indexing needed on each startup")


if __name__ == "__main__":
    main()