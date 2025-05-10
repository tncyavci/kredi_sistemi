import os
import pytest
import tempfile
import shutil
import pandas as pd
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from utils.preprocessing.enhanced_pdf_processor import EnhancedPdfProcessor
import utils.preprocessing.enhanced_pdf_processor

@pytest.fixture
def test_pdf_dir():
    """Test PDF dizini"""
    test_dir = Path(__file__).parent.parent / "test_pdfs"
    if not test_dir.exists() or len(list(test_dir.glob("*.pdf"))) < 2:
        pytest.skip("Test PDF'leri bulunamadı. Önce 'create_test_pdfs.py' çalıştırılmalı.")
    return str(test_dir)

@pytest.fixture
def output_dir():
    """Geçici çıktı dizini"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

class TestPdfProcessorIntegration:
    """EnhancedPdfProcessor'ın entegrasyon testleri"""
    
    def test_directory_processing(self, test_pdf_dir, output_dir):
        """PDF dizini işleme entegrasyon testi"""
        processor = EnhancedPdfProcessor(output_dir=output_dir)
        
        # Kategori eşleştirmeleri
        category_mapping = {
            "loan": "kredi",
            "mortgage": "ipotek",
            "credit": "kredi_karti"
        }
        
        # Dizini işle
        results = processor.process_pdf_directory(
            test_pdf_dir,
            category_mapping=category_mapping
        )
        
        # Sonuçlar doğru mu?
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Her PDF için çıktı dosyası oluşturuldu mu?
        pdf_files = [f for f in os.listdir(test_pdf_dir) if f.lower().endswith('.pdf')]
        for pdf_file in pdf_files:
            output_file = os.path.join(output_dir, f"{os.path.splitext(pdf_file)[0]}.json")
            assert os.path.exists(output_file)
            
            # JSON dosyası doğru mu?
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                assert "id" in data
                assert "text" in data
                assert "chunks" in data
    
    def test_streaming_processing(self, test_pdf_dir, output_dir):
        """Akış tabanlı PDF işleme entegrasyon testi"""
        processor = EnhancedPdfProcessor(
            output_dir=output_dir,
            use_streaming=True,
            memory_threshold_mb=100
        )
        
        # İlk PDF dosyasını akış tabanlı işle
        pdf_files = [f for f in os.listdir(test_pdf_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            pytest.skip("Test PDF'leri bulunamadı.")
            
        pdf_path = os.path.join(test_pdf_dir, pdf_files[0])
        
        # Akış tabanlı işleme
        result = processor.process_pdf_streaming(
            pdf_path=pdf_path,
            category="test"
        )
        
        # Sonuç doğru mu?
        assert isinstance(result, dict)
        assert "id" in result
        assert "processing_completed" in result
        assert result["processing_completed"] is True
        
        # Özet dosyası oluşturuldu mu?
        result_id = result["id"]
        summary_file = os.path.join(output_dir, f"{result_id}_summary.json")
        assert os.path.exists(summary_file)
    
    def test_parallel_processing(self, test_pdf_dir, output_dir):
        """Paralel PDF işleme entegrasyon testi"""
        # Birden fazla PDF dosyası olduğundan emin ol
        pdf_files = [f for f in os.listdir(test_pdf_dir) if f.lower().endswith('.pdf')]
        if len(pdf_files) < 2:
            pytest.skip("Test için en az 2 PDF dosyası gerekli.")
        
        processor = EnhancedPdfProcessor(
            output_dir=output_dir,
            max_workers=2  # Test için 2 işçi kullan
        )
        
        # Paralel işleme
        results = processor.process_pdf_directory_streaming(
            directory=test_pdf_dir,
            parallel=True
        )
        
        # Sonuçlar doğru mu?
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Her PDF için özet dosyası oluşturuldu mu?
        for result in results:
            result_id = result["id"]
            summary_file = os.path.join(output_dir, f"{result_id}_summary.json")
            assert os.path.exists(summary_file)
    
    def test_table_extraction(self, test_pdf_dir, output_dir):
        """Tablo çıkarma entegrasyon testi"""
        processor = EnhancedPdfProcessor(output_dir=output_dir)
        
        # Tablolu bir PDF bul (varsayımsal olarak mortgage_application.pdf içerisinde tablo vardır)
        pdf_path = os.path.join(test_pdf_dir, "mortgage_application.pdf")
        if not os.path.exists(pdf_path):
            pytest.skip("Tablo testi için 'mortgage_application.pdf' bulunamadı.")
        
        # Tablo çıkarma ile işle
        result = processor.process_pdf(
            pdf_path,
            category="test",
            extract_tables=True
        )
        
        # Tablolar çıkarıldı mı?
        assert "tables" in result
        
        # Not: Gerçek tablolar için daha kapsamlı testler test_pdfs dizinindeki
        # PDF dosyalarında gerçek tablolar olduğunda eklenebilir.
    
    def test_ocr_processing(self, test_pdf_dir, output_dir, monkeypatch):
        """OCR işleme entegrasyon testi"""
        # OCR'ın çağrıldığını izlemeye yarayan sahte fonksiyon
        ocr_called = False
        
        def mock_perform_ocr(*args, **kwargs):
            nonlocal ocr_called
            ocr_called = True
            return "OCR ile çıkarılan örnek metin"
        
        # EnhancedPdfProcessor.perform_ocr metodunu monkeypatch ile değiştir
        monkeypatch.setattr(EnhancedPdfProcessor, "perform_ocr", mock_perform_ocr)
        
        # PDF işlemci oluştur
        processor = EnhancedPdfProcessor(output_dir=output_dir)
        
        # OCR ile işle (sahte metin çıkarma sonuçları vermek için metin çıkarma fonksiyonlarını da değiştir)
        def mock_extract_text(*args, **kwargs):
            return ""  # Boş metin döndür ki OCR kullanılsın
        
        monkeypatch.setattr(processor, "extract_text_with_pymupdf", mock_extract_text)
        monkeypatch.setattr(processor, "extract_text_with_pdfplumber", mock_extract_text)
        
        # PDF'i işle (OCR etkin)
        pdf_files = [f for f in os.listdir(test_pdf_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            pytest.skip("Test PDF'leri bulunamadı.")
            
        pdf_path = os.path.join(test_pdf_dir, pdf_files[0])
        
        result = processor.process_pdf(
            pdf_path,
            use_ocr=True
        )
        
        # OCR fonksiyonu çağrıldı mı?
        assert ocr_called is True
    
    def test_memory_monitoring_integration(self, test_pdf_dir, output_dir, monkeypatch):
        """Bellek izleme entegrasyon testi"""
        # Bellek izleme fonksiyonunun çağrıldığını takip et
        check_memory_called = 0
        original_check_memory = EnhancedPdfProcessor._check_memory_usage
        
        def mock_check_memory(self):
            nonlocal check_memory_called
            check_memory_called += 1
            return original_check_memory(self)
        
        # Bellek izleme fonksiyonunu değiştir
        monkeypatch.setattr(EnhancedPdfProcessor, "_check_memory_usage", mock_check_memory)
        
        # Düşük bellek eşiği ile işlemci oluştur
        processor = EnhancedPdfProcessor(
            output_dir=output_dir,
            memory_threshold_mb=1  # Çok düşük bir eşik, bellek temizlemenin tetiklenmesi için
        )
        
        # PDF'i işle
        pdf_files = [f for f in os.listdir(test_pdf_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            pytest.skip("Test PDF'leri bulunamadı.")
            
        pdf_path = os.path.join(test_pdf_dir, pdf_files[0])
        
        result = processor.process_pdf(pdf_path)
        
        # Bellek izleme fonksiyonu çağrıldı mı?
        assert check_memory_called > 0
    
    def test_document_generation(self, test_pdf_dir):
        """Belge oluşturma entegrasyon testi"""
        # PDF'ten doğrudan belge oluşturma
        pdf_files = [f for f in os.listdir(test_pdf_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            pytest.skip("Test PDF'leri bulunamadı.")
            
        pdf_path = os.path.join(test_pdf_dir, pdf_files[0])
        
        documents = EnhancedPdfProcessor.process_pdf_to_documents(
            pdf_path,
            category="test",
            source="test_source"
        )
        
        # Belge listesi oluşturuldu mu?
        assert isinstance(documents, list)
        assert len(documents) > 0
        
        # Belge yapısı doğru mu?
        for doc in documents:
            assert "id" in doc
            assert "text" in doc
            assert "metadata" in doc
            assert doc["metadata"]["category"] == "test"
            assert doc["metadata"]["source"] == "test_source" 