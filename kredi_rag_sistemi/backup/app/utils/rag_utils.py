"""
RAG sistemi için yardımcı fonksiyonlar.
"""
import os
import logging
import gc
from typing import List, Dict, Any, Optional

from app.core.rag import KrediRAG
from models.llm import download_mistral_model, MistralLLM
from models.embeddings import DocumentEmbedder
from utils.preprocessing.pdf_processor import PdfProcessor
from app.config import (
    MODEL_CONFIG,
    PDF_CONFIG,
    VECTOR_DB_CONFIG,
    CATEGORY_MAPPING,
    SYSTEM_PROMPT
)

logger = logging.getLogger(__name__)

def setup_rag_system(
    model_dir: str,
    vector_db_path: str,
    model_path: Optional[str] = None
) -> KrediRAG:
    """
    RAG sistemini kurar ve başlatır.
    
    Args:
        model_dir: Model dosyalarının bulunduğu dizin
        vector_db_path: Vektör veritabanı dosya yolu
        model_path: Mistral model dosyasının yolu (belirtilmezse indirilir)
        
    Returns:
        Başlatılmış RAG sistemi
    """
    # Model yollarını belirle
    if not model_path:
        logger.info("Mistral modeli indiriliyor...")
        model_path = download_mistral_model(
            model_name=MODEL_CONFIG["llm_model"],
            file_name=MODEL_CONFIG["llm_file"],
            save_dir=model_dir
        )
        logger.info(f"Model indirildi: {model_path}")
    
    # RAG sistemini oluştur
    logger.info("RAG sistemi başlatılıyor...")
    rag = KrediRAG(
        model_path=model_path,
        vector_db_path=vector_db_path,
        embedding_model=MODEL_CONFIG["embedding_model"],
        top_k=VECTOR_DB_CONFIG["top_k"],
        chunk_size=PDF_CONFIG["chunk_size"],
        chunk_overlap=PDF_CONFIG["chunk_overlap"]
    )
    
    return rag

def process_single_pdf(
    pdf_path: str,
    rag: KrediRAG,
    category_mapping: Optional[Dict[str, str]] = None
) -> None:
    """
    Tek bir PDF dosyasını işler ve RAG sistemine ekler.
    
    Args:
        pdf_path: PDF dosya yolu
        rag: RAG sistemi örneği
        category_mapping: Kategori eşleştirme sözlüğü
    """
    filename = os.path.basename(pdf_path)
    
    # Dosyaya uygun kategoriyi belirle
    category = "finansal_rapor"  # Varsayılan kategori
    if category_mapping:
        for pattern, cat in category_mapping.items():
            if pattern.lower() in filename.lower():
                category = cat
                break
    
    # PDF'i işle ve belgeleri oluştur
    logger.info(f"PDF işleniyor: {filename}")
    documents = PdfProcessor.process_pdf_to_documents(
        pdf_path, 
        category=category,
        source=filename,
        chunk_size=PDF_CONFIG["chunk_size"],
        overlap=PDF_CONFIG["chunk_overlap"],
        max_pages=PDF_CONFIG["max_pages"]
    )
    
    if documents:
        # Belgeleri RAG sistemine ekle
        logger.info(f"{len(documents)} belge RAG sistemine ekleniyor...")
        rag.ingest_documents(documents)
        logger.info(f"{filename} için {len(documents)} belge eklendi.")
    else:
        logger.warning(f"{filename} için belge oluşturulamadı.")
    
    # Belleği temizle
    del documents
    gc.collect()

def process_pdf_directory(
    pdf_dir: str,
    rag: KrediRAG,
    category_mapping: Optional[Dict[str, str]] = None
) -> None:
    """
    Bir dizindeki tüm PDF dosyalarını işler.
    
    Args:
        pdf_dir: PDF dosyalarının bulunduğu dizin
        rag: RAG sistemi örneği
        category_mapping: Kategori eşleştirme sözlüğü
    """
    # PDF dosyalarını bul
    logger.info(f"PDF dizini işleniyor: {pdf_dir}")
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logger.warning(f"Dizinde PDF dosyası bulunamadı: {pdf_dir}")
        return
    
    # Her PDF dosyasını ayrı ayrı işle
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        process_single_pdf(pdf_path, rag, category_mapping)
        
        # Her PDF'den sonra belleği temizle
        gc.collect()
    
    # Vektör veritabanını kaydet
    rag.save_vector_db()
    logger.info("Tüm PDF'ler işlendi ve veritabanı kaydedildi.")

def query_rag_system(
    rag: KrediRAG,
    query: str,
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    RAG sistemine sorgu gönderir ve yanıt alır.
    
    Args:
        rag: RAG sistemi örneği
        query: Kullanıcı sorgusu
        system_prompt: Sistem promptu (belirtilmezse varsayılan kullanılır)
        
    Returns:
        Sorgu yanıtı ve ilgili belgeler
    """
    system_prompt = system_prompt or SYSTEM_PROMPT
    response = rag.query(query, system_prompt)
    
    return {
        "query": query,
        "response": response["response"],
        "relevant_documents": response["relevant_documents"]
    } 