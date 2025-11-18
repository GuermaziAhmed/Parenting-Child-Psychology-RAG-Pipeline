"""Process PDFs and text files from data/ directory using LangChain.

Extracts text from PDFs with pdfplumber, loads .txt files, cleans text,
chunks into semantic units with RecursiveCharacterTextSplitter, and returns
LangChain Document objects ready for embedding/vector store ingestion.

Usage:
    python process_pdfs.py

Returns a list of Document objects with metadata (source file, chunk index).
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import List, Dict, Any

try:
    import pdfplumber
except ImportError:
    print("pdfplumber not installed. Run: pip install pdfplumber")
    pdfplumber = None

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

print(f"Using CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP}")
def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file using pdfplumber."""
    if pdfplumber is None:
        raise ImportError("pdfplumber is required for PDF processing")
    
    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""
    
    return "\n\n".join(text_parts)


def load_text_file(txt_path: Path) -> str:
    """Load text from a .txt file with UTF-8 encoding."""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading text from {txt_path}: {e}")
        return ""


def clean_text(text: str) -> str:
    """Clean extracted text: normalize whitespace, remove artifacts, etc."""
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove common PDF artifacts (optional, adjust as needed)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def load_all_documents() -> List[Document]:
    """Load all PDFs and text files from DATA_DIR, clean, and return as Documents."""
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        return []
    
    documents = []
    
    # Load PDFs
    for pdf_file in DATA_DIR.glob("*.pdf"):
        print(f"Processing PDF: {pdf_file.name}")
        raw_text = extract_text_from_pdf(pdf_file)
        if raw_text:
            cleaned = clean_text(raw_text)
            if len(cleaned) > 100:  # Skip nearly empty docs
                doc = Document(
                    page_content=cleaned,
                    metadata={"source": pdf_file.name, "type": "pdf"}
                )
                documents.append(doc)
    
    # Load text files
    for txt_file in DATA_DIR.glob("*.txt"):
        print(f"Processing text: {txt_file.name}")
        raw_text = load_text_file(txt_file)
        if raw_text:
            cleaned = clean_text(raw_text)
            if len(cleaned) > 100:
                doc = Document(
                    page_content=cleaned,
                    metadata={"source": txt_file.name, "type": "text"}
                )
                documents.append(doc)
    
    print(f"\nLoaded {len(documents)} documents from {DATA_DIR}")
    return documents


def chunk_documents(documents: List[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """Split documents into smaller chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")
    
    # Add chunk index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    
    return chunks
import csv

def save_chunks_to_csv(chunks, output_path="chunks.csv"):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_id", "source", "type", "content"])
        for c in chunks:
            writer.writerow([
                c.metadata.get("chunk_id"),
                c.metadata.get("source"),
                c.metadata.get("type"),
                c.page_content
            ])
    print(f"Saved {len(chunks)} chunks to {output_path}")
def save_chunks_to_jsonl(chunks, output_path="chunks.jsonl"):
    import json
    with open(output_path, "w", encoding="utf-8") as f:
        for c in chunks:
            record = {
                "chunk_id": c.metadata.get("chunk_id"),
                "source": c.metadata.get("source"),
                "type": c.metadata.get("type"),
                "content": c.page_content
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(chunks)} chunks to {output_path}")




def main():
    """Main processing pipeline: load documents, clean, chunk, and return."""
    print("=== PDF & Text Processing Pipeline ===\n")
    
    # Load all documents
    documents = load_all_documents()
    if not documents:
        print("No documents found or loaded.")
        return []
    
    # Chunk documents
    chunks = chunk_documents(documents)
    
    # Display sample
    if chunks:
        print("\n=== Sample Chunk ===")
        sample = chunks[0]
        print(f"Source: {sample.metadata.get('source')}")
        print(f"Type: {sample.metadata.get('type')}")
        print(f"Chunk ID: {sample.metadata.get('chunk_id')}")
        print(f"Content preview (first 300 chars):\n{sample.page_content[:300]}...")
    
    return chunks


if __name__ == "__main__":
    chunks = main()
    print(f"\n✅ Processed {len(chunks)} total chunks ready for vector store.")
    #save_chunks_to_csv(chunks, output_path="chunks.csv")
    save_chunks_to_jsonl(chunks, output_path="chunks.jsonl")
    
    #print(chunks[:5])