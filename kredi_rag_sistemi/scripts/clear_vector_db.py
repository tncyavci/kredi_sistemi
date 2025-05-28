from app.core.rag import KrediRAG
import os
from pathlib import Path

# Get default vector DB path
base_dir = Path(__file__).parent
vector_db_path = str(base_dir / "data" / "vector_db")

# Get model path
model_path = str(base_dir / "models" / "mistral-7b-instruct-v0.2.Q4_K_M.gguf")

# Create a new RAG instance
print("Initializing RAG system...")
rag = KrediRAG(vector_db_path=vector_db_path, model_path=model_path)

# Clear the vector database
print("Clearing vector database...")
rag.clear_vector_db()
print("Vector database has been successfully cleared!") 