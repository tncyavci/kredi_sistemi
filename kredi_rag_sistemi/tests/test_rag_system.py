import os
import pytest
from pathlib import Path
import tempfile
import shutil

from app.core.rag import KrediRAG
from models.llm import MistralLLM
from models.embeddings import DocumentEmbedder
from app.config import (
    MODEL_CONFIG,
    PDF_CONFIG,
    VECTOR_DB_CONFIG,
    SYSTEM_PROMPT
)

@pytest.fixture
def temp_model_dir():
    """Test için geçici model dizini oluşturur"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def temp_vector_db_path():
    """Test için geçici vektör veritabanı yolu oluşturur"""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_file:
        return temp_file.name

@pytest.fixture
def sample_documents():
    """Test için örnek belgeler oluşturur"""
    return [
        {
            "id": "doc1",
            "text": "Bu bir test belgesidir. Finansal veriler içerir.",
            "metadata": {
                "category": "test",
                "source": "test.pdf"
            }
        },
        {
            "id": "doc2",
            "text": "Bu başka bir test belgesidir. Mali tablo bilgileri içerir.",
            "metadata": {
                "category": "test",
                "source": "test2.pdf"
            }
        }
    ]

@pytest.fixture
def rag_system(temp_model_dir, temp_vector_db_path):
    """Test için RAG sistemi örneği oluşturur"""
    # Model dosyasını kopyala
    model_path = os.path.join(temp_model_dir, MODEL_CONFIG["llm_file"])
    shutil.copy(
        os.path.join(Path(__file__).parent.parent, "models", MODEL_CONFIG["llm_file"]),
        model_path
    )
    
    # RAG sistemini oluştur
    rag = KrediRAG(
        model_path=model_path,
        vector_db_path=temp_vector_db_path,
        embedding_model=MODEL_CONFIG["embedding_model"],
        top_k=VECTOR_DB_CONFIG["top_k"],
        chunk_size=PDF_CONFIG["chunk_size"],
        chunk_overlap=PDF_CONFIG["chunk_overlap"]
    )
    
    return rag

def test_rag_initialization(rag_system):
    """RAG sistemi başlatma testi"""
    assert isinstance(rag_system, KrediRAG)
    assert isinstance(rag_system.llm, MistralLLM)
    assert isinstance(rag_system.embedder, DocumentEmbedder)
    assert rag_system.top_k == VECTOR_DB_CONFIG["top_k"]

def test_document_ingestion(rag_system, sample_documents):
    """Belge ekleme testi"""
    # Belgeleri ekle
    rag_system.ingest_documents(sample_documents)
    
    # Vektör veritabanını kaydet
    rag_system.save_vector_db()
    
    # Vektör veritabanını yükle
    rag_system.load_vector_db()
    
    # Belgelerin eklendiğini kontrol et
    assert len(rag_system.embedder.vector_db) > 0

def test_query_processing(rag_system, sample_documents):
    """Sorgu işleme testi"""
    # Belgeleri ekle
    rag_system.ingest_documents(sample_documents)
    
    # Test sorgusu
    query = "Finansal veriler nelerdir?"
    response = rag_system.query(query, SYSTEM_PROMPT)
    
    # Yanıt kontrolü
    assert "query" in response
    assert "response" in response
    assert "relevant_documents" in response
    assert len(response["relevant_documents"]) > 0
    
    # İlgili belgelerin kontrolü
    for doc in response["relevant_documents"]:
        assert "id" in doc
        assert "score" in doc
        assert "text" in doc
        assert "metadata" in doc

def test_system_prompt_handling(rag_system, sample_documents):
    """Sistem promptu işleme testi"""
    # Belgeleri ekle
    rag_system.ingest_documents(sample_documents)
    
    # Özel sistem promptu
    custom_prompt = "Sen bir test asistanısın."
    query = "Test sorgusu"
    response = rag_system.query(query, custom_prompt)
    
    assert "response" in response
    assert response["response"] is not None

def test_error_handling(rag_system):
    """Hata yönetimi testi"""
    # Geçersiz sorgu
    response = rag_system.query("")
    assert response["response"] is not None
    
    # Geçersiz belge
    invalid_docs = [{"id": "invalid", "text": ""}]
    rag_system.ingest_documents(invalid_docs)
    assert len(rag_system.embedder.vector_db) == 0 