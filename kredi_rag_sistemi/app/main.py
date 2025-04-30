import os
import sys
import logging
from pathlib import Path

# Ana dizini ekle (geliştirme ortamına göre ayarlayın)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from app.core.rag import KrediRAG
from app.core.pdf_processor import PDFProcessor
from models.llm import download_mistral_model, MistralLLM
from models.embeddings import DocumentEmbedder

def setup_dirs():
    """Gerekli dizinleri oluşturur"""
    os.makedirs(os.path.join(ROOT_DIR, "data", "processed"), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "data", "raw"), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "models", "embeddings"), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "logs"), exist_ok=True)

# Önce dizinleri oluştur
setup_dirs()

# Loglama ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(ROOT_DIR, "logs", "app.log"), mode="a")
    ]
)

logger = logging.getLogger(__name__)

def process_pdf_documents():
    """PDF belgelerini işler"""
    pdf_dir = os.path.join(ROOT_DIR, "..", "pdf")  # PDF'lerin bulunduğu dizin
    processor = PDFProcessor(pdf_dir)
    return processor.process_pdfs()

def main():
    """Ana uygulama fonksiyonu"""
    # Model yollarını belirle
    model_dir = os.path.join(ROOT_DIR, "models")
    vector_db_path = os.path.join(model_dir, "embeddings", "vector_db.pkl")
    
    # PDF belgelerini işle
    logger.info("PDF belgeleri işleniyor...")
    documents = process_pdf_documents()
    
    # Mistral modelini indir
    logger.info("Mistral modeli indiriliyor (bu işlem birkaç dakika sürebilir)...")
    model_path = download_mistral_model(save_dir=model_dir)
    logger.info(f"Model indirildi: {model_path}")
    
    # RAG sistemini oluştur
    logger.info("RAG sistemi başlatılıyor...")
    rag = KrediRAG(
        model_path=model_path,
        vector_db_path=vector_db_path,
        top_k=3
    )
    
    # Belgeleri ekle
    logger.info("PDF belgeleri RAG sistemine ekleniyor...")
    rag.ingest_documents(documents)
    
    # Örnek sorgular
    sample_queries = [
        "Adel Mali Tablo ve Dipnotlar hakkında bilgi verir misiniz?",
        "Adel Kalemcilik 2024 Faaliyet Raporunda neler var?",
        "KuzeyKablo 2023 fiyatları nasıl?",
        "Pegasus'un özel finansal bilgileri neler?"
    ]
    
    # Sorguları işle ve sonuçları göster
    logger.info("Sorgular işleniyor...")
    for i, query in enumerate(sample_queries, 1):
        print(f"\n{'='*80}\nSORGU {i}: {query}\n{'='*80}")
        
        response = rag.query(query)
        
        print(f"\nCEVAP:\n{response['response']}\n")
        print(f"KULLANILAN BELGELER:")
        for idx, doc in enumerate(response['relevant_documents'][:3], 1):
            print(f"\nBelge {idx} (Benzerlik: {doc['score']:.4f}):")
            print(f"Kategori: {doc['metadata'].get('category', 'Belirtilmemiş')}")
            print(f"Kaynak: {doc['metadata'].get('kaynak', 'Belirtilmemiş')}")
            print(f"İçerik: {doc['text'][:150]}...")
    
    logger.info("RAG demo tamamlandı.")

if __name__ == "__main__":
    main() 