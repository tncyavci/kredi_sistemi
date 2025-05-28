import os
import sys
import uvicorn

# Ana dizini ekle
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

def main():
    """FastAPI uygulamasını çalıştırır"""
    print("Kredi RAG API başlatılıyor...")
    uvicorn.run(
        "app.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    main() 