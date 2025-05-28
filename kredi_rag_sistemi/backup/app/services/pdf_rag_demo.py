import os
import sys
import logging
import argparse
from pathlib import Path

# Ana dizini ekle
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from app.utils.rag_utils import (
    setup_rag_system,
    process_pdf_directory,
    query_rag_system
)
from app.config import (
    CATEGORY_MAPPING,
    SAMPLE_QUERIES,
    SYSTEM_PROMPT
)

# Dizinleri oluştur
os.makedirs(os.path.join(ROOT_DIR, "logs"), exist_ok=True)

# Loglama ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(ROOT_DIR, "logs", "pdf_rag.log"), mode="a")
    ]
)

logger = logging.getLogger(__name__)

def run_sample_queries(rag):
    """Örnek sorguları çalıştırır ve sonuçları gösterir"""
    logger.info("Örnek sorgular işleniyor...")
    for i, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"\n{'='*80}\nSORGU {i}: {query}\n{'='*80}")
        
        response = query_rag_system(rag, query, SYSTEM_PROMPT)
        
        print(f"\nCEVAP:\n{response['response']}\n")
        print(f"KULLANILAN BELGELER:")
        for idx, doc in enumerate(response['relevant_documents'][:3], 1):
            print(f"\nBelge {idx} (Benzerlik: {doc['score']:.4f}):")
            print(f"Kaynak: {doc['metadata'].get('source', 'Belirtilmemiş')}")
            print(f"Kategori: {doc['metadata'].get('category', 'Belirtilmemiş')}")
            print(f"Konum: {doc['metadata'].get('page_range', 'Belirtilmemiş')}")
            print(f"İçerik: {doc['text'][:150]}...")

def run_interactive_mode(rag):
    """Etkileşimli modu çalıştırır"""
    print("\n\n" + "="*80)
    print("Etkileşimli Mod: Sorgunuzu yazın (çıkmak için 'q' veya 'quit')")
    print("="*80 + "\n")
    
    while True:
        user_query = input("Sorgu: ")
        if user_query.lower() in ['q', 'quit', 'exit', 'çıkış']:
            break
            
        response = query_rag_system(rag, user_query, SYSTEM_PROMPT)
        print(f"\nCEVAP:\n{response['response']}\n")

def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description="PDF dosyalarından RAG oluşturma")
    parser.add_argument("--pdf_dir", type=str, help="PDF dosyalarının bulunduğu dizin")
    parser.add_argument("--interactive", action="store_true", help="Etkileşimli mod açık")
    args = parser.parse_args()
    
    # PDF dizinini kontrol et
    pdf_dir = args.pdf_dir
    if not pdf_dir or not os.path.exists(pdf_dir):
        logger.error(f"Geçerli bir PDF dizini belirtilmeli: {pdf_dir}")
        return
    
    # Model yollarını belirle
    model_dir = os.path.join(ROOT_DIR, "models")
    vector_db_path = os.path.join(model_dir, "embeddings", "pdf_vector_db.pkl")
    
    # RAG sistemini oluştur
    rag = setup_rag_system(model_dir, vector_db_path)
    
    # PDF'leri işle
    process_pdf_directory(pdf_dir, rag, CATEGORY_MAPPING)
    
    # Örnek sorguları çalıştır
    run_sample_queries(rag)
    
    # Etkileşimli mod
    if args.interactive:
        run_interactive_mode(rag)
    
    logger.info("PDF RAG demo tamamlandı.")

if __name__ == "__main__":
    main() 