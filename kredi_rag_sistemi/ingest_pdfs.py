import os
from pathlib import Path
from app.core.rag import KrediRAG
from app.services.pdf_processor import PDFProcessor

def main():
    # Get paths
    base_dir = Path(__file__).parent
    model_path = str(base_dir / "models" / "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    vector_db_path = str(base_dir / "data" / "vector_db")
    pdf_dir = str(base_dir / "test_pdfs")
    
    print(f"PDF directory: {pdf_dir}")
    print(f"Vector DB path: {vector_db_path}")
    
    # Create RAG instance
    print("Initializing RAG system...")
    rag = KrediRAG(
        vector_db_path=vector_db_path,
        model_path=model_path
    )
    
    # Create PDF processor
    print("Creating PDF processor...")
    processor = PDFProcessor(
        input_dir=pdf_dir,
        use_ocr=False,  # OCR is problematic, disabling
        extract_tables=False  # Disable tables to avoid the error
    )
    
    # Process PDFs
    print("Processing PDFs...")
    documents = processor.process_pdfs()
    
    print(f"Processed {len(documents)} documents")
    
    # Ingest documents into RAG
    if documents:
        print(f"Ingesting {len(documents)} documents into RAG system...")
        rag.ingest_documents(documents)
        print("PDFs have been successfully processed and added to the vector database!")
    else:
        print("No documents were processed. Check if there are PDFs in the directory.")
    
    # Get document count
    doc_count = rag.get_document_count()
    print(f"Total document chunks in vector database: {doc_count}")

if __name__ == "__main__":
    main() 