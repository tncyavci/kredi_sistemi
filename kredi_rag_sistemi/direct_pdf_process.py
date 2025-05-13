import os
from pathlib import Path
from utils.preprocessing.enhanced_pdf_processor import EnhancedPdfProcessor
from app.core.rag import KrediRAG

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
    
    # Process PDFs directly using static method
    documents = []
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    print(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"Processing {pdf_file}...")
        
        try:
            # Process PDF directly with static method
            pdf_documents = EnhancedPdfProcessor.process_pdf_to_documents(
                pdf_path=pdf_path,
                category="finansal_bilgiler",
                source=pdf_file
            )
            
            documents.extend(pdf_documents)
            print(f"Created {len(pdf_documents)} document chunks from {pdf_file}")
            
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
    
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