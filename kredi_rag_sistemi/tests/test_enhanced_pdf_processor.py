import os
import pytest
import tempfile
import json
import pandas as pd
from pathlib import Path

from utils.preprocessing.enhanced_pdf_processor import EnhancedPdfProcessor

@pytest.fixture
def sample_pdf_path():
    """Test için örnek PDF dosyasının yolunu döndürür"""
    test_dir = Path(__file__).parent.parent / "test_pdfs"
    if not test_dir.exists():
        pytest.skip("Test PDF'leri bulunamadı. Önce 'create_test_pdfs.py' çalıştırılmalı.")
    return str(test_dir / "simple_loan.pdf")

@pytest.fixture
def complex_pdf_path():
    """Tablolar içeren karmaşık örnek PDF dosyasının yolunu döndürür"""
    test_dir = Path(__file__).parent.parent / "test_pdfs"
    if not test_dir.exists():
        pytest.skip("Test PDF'leri bulunamadı. Önce 'create_test_pdfs.py' çalıştırılmalı.")
    return str(test_dir / "mortgage_application.pdf")

@pytest.fixture
def enhanced_processor():
    """EnhancedPdfProcessor örneği"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield EnhancedPdfProcessor(output_dir=tmp_dir)

def test_pdf_validation(enhanced_processor, sample_pdf_path):
    """PDF dosya doğrulama testi"""
    # Geçerli PDF
    assert enhanced_processor._validate_pdf_file(sample_pdf_path) is True
    
    # Geçersiz dosya yolu
    assert enhanced_processor._validate_pdf_file("nonexistent.pdf") is False
    
    # Geçersiz uzantı
    invalid_path = sample_pdf_path.replace(".pdf", ".txt")
    assert enhanced_processor._validate_pdf_file(invalid_path) is False

def test_text_extraction_methods(enhanced_processor, sample_pdf_path):
    """Farklı metin çıkarma yöntemlerinin testi"""
    # PyMuPDF ile çıkarma
    pymupdf_text = enhanced_processor.extract_text_with_pymupdf(sample_pdf_path)
    assert isinstance(pymupdf_text, str)
    assert len(pymupdf_text) > 0
    
    # pdfplumber ile çıkarma
    pdfplumber_text = enhanced_processor.extract_text_with_pdfplumber(sample_pdf_path)
    assert isinstance(pdfplumber_text, str)
    assert len(pdfplumber_text) > 0

def test_text_cleaning(enhanced_processor):
    """Metin temizleme işlevlerinin testi"""
    # Birden fazla boşluk temizleme
    text = "  Birden    fazla   boşluk   "
    assert enhanced_processor._remove_extra_spaces(text) == "Birden fazla boşluk"
    
    # Özel karakter temizleme
    text = "Örnek metin: @#$%^&*()"
    assert "Örnek metin" in enhanced_processor._remove_special_characters(text)
    
    # Türkçe karakterlerin korunup korunmadığı
    text = "İşÇğĞüÜöÖşŞı"
    cleaned = enhanced_processor._remove_special_characters(text)
    assert cleaned == text
    
    # Boşluk normalleştirme
    text = "  Satır1\nSatır2  \n  Satır3  "
    assert enhanced_processor._normalize_whitespace(text) == "Satır1 Satır2 Satır3"
    
    # Tam temizleme işlemi
    text = "  Örnek   metin: @#$%^&*() \n\n yeni satır  "
    cleaned = enhanced_processor.clean_text(text)
    assert isinstance(cleaned, str)
    assert len(cleaned) > 0
    assert "@#$%^&*()" not in cleaned

def test_metadata_extraction(enhanced_processor, sample_pdf_path):
    """Meta veri çıkarma testi"""
    metadata = enhanced_processor.extract_metadata(sample_pdf_path)
    
    assert isinstance(metadata, dict)
    assert "filename" in metadata
    assert "file_size_kb" in metadata
    assert "total_pages" in metadata
    assert metadata["filename"] == os.path.basename(sample_pdf_path)
    assert metadata["file_size_kb"] > 0
    assert metadata["total_pages"] > 0

def test_text_chunking(enhanced_processor):
    """Metin parçalama işlevinin testi"""
    sample_text = "Bu bir test metnidir. " * 100
    
    # Parçalama parametreleri
    chunk_size = 50
    overlap = 10
    
    chunks = enhanced_processor.chunk_text(sample_text, chunk_size, overlap)
    
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    
    # Her parçada gerekli alanlar var mı?
    for chunk in chunks:
        assert "text" in chunk
        assert "length" in chunk
        assert "byte_size" in chunk
        
    # Parça boyutu kontrolü
    for chunk in chunks:
        assert chunk["length"] <= chunk_size
        
    # Örtüşme kontrolü (en az iki parça varsa)
    if len(chunks) > 1:
        first_chunk_words = chunks[0]["text"].split()
        second_chunk_words = chunks[1]["text"].split()
        
        # İkinci parça, birinci parçanın son kelimelerini içermeli
        overlap_words = first_chunk_words[-overlap:]
        for word in overlap_words:
            assert word in second_chunk_words

def test_basic_pdf_processing(enhanced_processor, sample_pdf_path):
    """Temel PDF işleme testi"""
    result = enhanced_processor.process_pdf(
        sample_pdf_path,
        category="test",
        extract_tables=True,
        use_ocr=False
    )
    
    assert isinstance(result, dict)
    assert "id" in result
    assert "text" in result
    assert "metadata" in result
    assert "chunks" in result
    assert len(result["chunks"]) > 0
    assert result["total_words"] > 0
    assert result["category"] == "test"

def test_memory_usage_monitoring():
    """Bellek kullanımı izleme testi"""
    from utils.preprocessing.enhanced_pdf_processor import get_memory_usage
    
    # Bellek kullanımı bir sayı dönmeli
    memory_usage = get_memory_usage()
    assert isinstance(memory_usage, float)
    assert memory_usage > 0

def test_garbage_collection():
    """Çöp toplama fonksiyonlarının testi"""
    from utils.preprocessing.enhanced_pdf_processor import force_gc_after
    
    # Test fonksiyonu
    @force_gc_after
    def test_function():
        # Büyük liste oluştur ve döndür
        return [i for i in range(1000000)]
    
    # Fonksiyon çalışmalı ve değer dönmeli
    result = test_function()
    assert len(result) == 1000000 