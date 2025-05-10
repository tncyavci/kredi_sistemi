import os
import sys
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ana dizini ekle
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from app.api.router import router
from app.core.rag import KrediRAG
from models.llm import download_mistral_model

# Loglama ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(ROOT_DIR, "logs", "api.log"), mode="a")
    ]
)

logger = logging.getLogger(__name__)

# Global RAG instance'ı
_rag_instance = None

# FastAPI uygulaması
app = FastAPI(
    title="Kredi RAG API",
    description="Kredi ve finans belgeleri için RAG tabanlı soru-cevap API'si",
    version="1.0.0",
)

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Üretimde bunu kısıtlayın
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router ekle
app.include_router(router)

def initialize_rag():
    """RAG sistemini başlatır"""
    global _rag_instance
    
    try:
        # Gerekli dizinleri oluştur
        os.makedirs(os.path.join(ROOT_DIR, "data", "processed"), exist_ok=True)
        os.makedirs(os.path.join(ROOT_DIR, "models", "embeddings"), exist_ok=True)
        os.makedirs(os.path.join(ROOT_DIR, "logs"), exist_ok=True)
        
        # Model yollarını belirle
        model_dir = os.path.join(ROOT_DIR, "models")
        vector_db_path = os.path.join(model_dir, "embeddings", "vector_db.pkl")
        
        # Mistral modelini indir
        logger.info("Mistral modeli kontrol ediliyor...")
        model_path = download_mistral_model(save_dir=model_dir)
        logger.info(f"Model yolu: {model_path}")
        
        # RAG sistemini oluştur
        logger.info("RAG sistemi başlatılıyor...")
        _rag_instance = KrediRAG(
            model_path=model_path,
            vector_db_path=vector_db_path,
            top_k=3
        )
        
        # Daha önce işlenmiş veri var mı kontrol et
        if os.path.exists(vector_db_path):
            logger.info("Vektör veritabanı yüklendi")
        else:
            logger.warning("Vektör veritabanı bulunamadı - PDF'ler henüz işlenmemiş olabilir")
        
        logger.info("RAG sistemi başarıyla başlatıldı")
        return _rag_instance
    except Exception as e:
        logger.error(f"RAG başlatma hatası: {str(e)}")
        _rag_instance = None
        raise

# RAG singleton'ını dışa aktar
def get_rag_instance():
    if _rag_instance is None:
        initialize_rag()
    return _rag_instance

# Başlangıçta RAG sistemini başlat
@app.on_event("startup")
async def startup_event():
    try:
        initialize_rag()
    except Exception as e:
        logger.error(f"Uygulama başlatma hatası: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 