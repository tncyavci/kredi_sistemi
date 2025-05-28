import os
import pytest
from pathlib import Path
from utils.preprocessing.pdf_processor import PdfProcessor

@pytest.fixture
def sample_pdf_path():
    """Test için örnek PDF dosyasının yolunu döndürür"""
    test_dir = Path(__file__).parent.parent / "test_pdfs"
    return str(test_dir / "sample.pdf")

@pytest.fixture
def sample_text():
    """Test için örnek metin döndürür"""
    return "Bu bir test metnidir. Bu metin, PDF işleme testleri için kullanılacaktır. " * 10

def test_validate_pdf_file(sample_pdf_path):
    """PDF dosya doğrulama testi"""
    # Geçerli PDF
    assert PdfProcessor._validate_pdf_file(sample_pdf_path) is True
    
    # Geçersiz dosya yolu
    assert PdfProcessor._validate_pdf_file("nonexistent.pdf") is False
    
    # Geçersiz uzantı
    invalid_path = sample_pdf_path.replace(".pdf", ".txt")
    assert PdfProcessor._validate_pdf_file(invalid_path) is False

def test_extract_text_from_pdf(sample_pdf_path):
    """PDF'ten metin çıkarma testi"""
    text = PdfProcessor.extract_text_from_pdf(sample_pdf_path)
    assert isinstance(text, str)
    assert len(text) > 0
    
    # Sayfa sınırlaması testi
    text_limited = PdfProcessor.extract_text_from_pdf(sample_pdf_path, max_pages=1)
    assert len(text_limited) < len(text)

def test_chunk_text(sample_text):
    """Metin parçalama testi"""
    chunks = PdfProcessor.chunk_text(sample_text, chunk_size=100, overlap=20)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    
    # Parça boyutu kontrolü
    for chunk in chunks:
        assert len(chunk) <= 100
    
    # Örtüşme kontrolü
    for i in range(len(chunks) - 1):
        overlap_text = chunks[i][-20:]
        assert overlap_text in chunks[i + 1]

def test_process_pdf_to_documents(sample_pdf_path):
    """PDF'ten belge oluşturma testi"""
    documents = PdfProcessor.process_pdf_to_documents(
        sample_pdf_path,
        category="test",
        source="sample.pdf",
        chunk_size=100,
        overlap=20
    )
    
    assert isinstance(documents, list)
    assert len(documents) > 0
    
    # Belge yapısı kontrolü
    for doc in documents:
        assert "id" in doc
        assert "text" in doc
        assert "metadata" in doc
        assert doc["metadata"]["category"] == "test"
        assert doc["metadata"]["source"] == "sample.pdf"

def test_process_pdf_directory(tmp_path, sample_pdf_path):
    """PDF dizini işleme testi"""
    # Test dizini oluştur
    test_dir = tmp_path / "test_pdfs"
    test_dir.mkdir()
    
    # Örnek PDF'i kopyala
    import shutil
    shutil.copy(sample_pdf_path, test_dir)
    
    # Dizini işle
    documents = PdfProcessor.process_pdf_directory(
        str(test_dir),
        category_mapping={"sample": "test"},
        chunk_size=100,
        overlap=20
    )
    
    assert isinstance(documents, list)
    assert len(documents) > 0

def test_pdf_processor_initialization():
    processor = PDFProcessor("test_pdfs")
    assert processor.pdf_dir == "test_pdfs"
    assert len(processor.text_cleaners) == 3

def test_text_cleaning():
    processor = PDFProcessor("test_pdfs")
    
    # Test extra spaces
    text = "  multiple   spaces  "
    assert processor._remove_extra_spaces(text) == "multiple spaces"
    
    # Test special characters
    text = "Hello! @#$%^&*()"
    assert processor._remove_special_characters(text) == "Hello"
    
    # Test whitespace normalization
    text = "  line1\nline2  \n  line3  "
    assert processor._normalize_whitespace(text) == "line1 line2 line3"

def test_metadata_extraction():
    processor = PDFProcessor("test_pdfs")
    # Bu test için gerçek bir PDF dosyası gerekiyor
    # Test PDF'i oluşturulduktan sonra bu test güncellenecek

def test_process_pdf():
    processor = PDFProcessor("test_pdfs")
    # Bu test için gerçek bir PDF dosyası gerekiyor
    # Test PDF'i oluşturulduktan sonra bu test güncellenecek

def test_process_all_pdfs():
    processor = PDFProcessor("test_pdfs")
    # Bu test için gerçek PDF dosyaları gerekiyor
    # Test PDF'leri oluşturulduktan sonra bu test güncellenecek 