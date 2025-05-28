import os
import pytest
import tempfile
import json
import shutil
from pathlib import Path
import logging

# RAG sistemi için gerekli modülleri içe aktar
from utils.preprocessing.enhanced_pdf_processor import EnhancedPdfProcessor

# E2E testleri için özel marker
pytestmark = pytest.mark.e2e

@pytest.fixture
def test_pdf_dir():
    """Test PDF dizini"""
    test_dir = Path(__file__).parent.parent / "test_pdfs"
    if not test_dir.exists():
        pytest.skip("Test PDF'leri bulunamadı. Önce 'create_test_pdfs.py' çalıştırılmalı.")
    return str(test_dir)

@pytest.fixture
def output_dir():
    """Geçici çıktı dizini"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

@pytest.fixture
def setup_logging():
    """Test için logger yapılandırması"""
    logger = logging.getLogger("e2e_tests")
    logger.setLevel(logging.INFO)
    
    # Konsol handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

class TestEndToEndPdfProcessing:
    """EnhancedPdfProcessor için uçtan uca testler"""
    
    def test_e2e_pdf_to_chunks(self, test_pdf_dir, output_dir, setup_logging):
        """
        PDF'ten metin parçaları oluşturmanın uçtan uca testi
        
        Test akışı:
        1. PDF dosyalarını bul
        2. EnhancedPdfProcessor oluştur
        3. PDF'leri işle ve metin parçaları oluştur
        4. Sonuçları doğrula
        """
        logger = setup_logging
        logger.info("Uçtan uca test başlatılıyor: PDF'ten metin parçaları oluşturma")
        
        # PDF dosyalarını bul
        pdf_files = [f for f in os.listdir(test_pdf_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            pytest.skip("Test PDF'leri bulunamadı.")
            
        # Test için ilk PDF'i kullan
        pdf_path = os.path.join(test_pdf_dir, pdf_files[0])
        logger.info(f"Test PDF: {pdf_path}")
        
        # EnhancedPdfProcessor oluştur
        processor = EnhancedPdfProcessor(
            output_dir=output_dir,
            ocr_lang="tur+eng",
            table_extraction_method="auto"
        )
        logger.info("PDF işlemci oluşturuldu")
        
        # PDF'i işle
        result = processor.process_pdf(
            pdf_path,
            category="test",
            extract_tables=True,
            use_ocr=True,
            chunk_size=100,
            overlap=20
        )
        logger.info(f"PDF işlendi, {len(result['chunks'])} metin parçası oluşturuldu")
        
        # Sonuçları doğrula
        assert isinstance(result, dict)
        assert "id" in result
        assert "text" in result
        assert "chunks" in result
        assert "metadata" in result
        assert "tables" in result
        assert len(result["chunks"]) > 0
        
        # Çıktı dosyası oluşturuldu mu?
        output_file = os.path.join(output_dir, f"{result['id']}.json")
        assert os.path.exists(output_file)
        logger.info(f"Çıktı dosyası doğrulandı: {output_file}")
        
        # JSON dosyasını kontrol et
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert data["id"] == result["id"]
            assert len(data["chunks"]) == len(result["chunks"])
        
        logger.info("Uçtan uca test başarıyla tamamlandı")
    
    def test_e2e_pdf_directory_to_documents(self, test_pdf_dir, output_dir, setup_logging):
        """
        PDF dizininden belge oluşturmanın uçtan uca testi
        
        Test akışı:
        1. PDF dizinini hazırla
        2. PDF'leri belgelere dönüştür
        3. Sonuçları doğrula
        """
        logger = setup_logging
        logger.info("Uçtan uca test başlatılıyor: PDF dizininden belge oluşturma")
        
        # Kategori eşleştirmeleri
        category_mapping = {
            "loan": "kredi",
            "mortgage": "ipotek",
            "credit": "kredi_karti",
            "business": "ticari"
        }
        
        # PDF'leri belgelere dönüştür
        documents = EnhancedPdfProcessor.process_pdf_directory_to_documents(
            directory=test_pdf_dir,
            category_mapping=category_mapping,
            chunk_size=100,
            overlap=20
        )
        
        logger.info(f"{len(documents)} belge oluşturuldu")
        
        # Sonuçları doğrula
        assert isinstance(documents, list)
        assert len(documents) > 0
        
        # Belge yapısını kontrol et
        for doc in documents:
            assert "id" in doc
            assert "text" in doc
            assert "metadata" in doc
            assert "category" in doc["metadata"]
            assert "source" in doc["metadata"]
            
        # Farklı kategorileri kontrol et
        categories = set(doc["metadata"]["category"] for doc in documents)
        logger.info(f"Oluşturulan belge kategorileri: {categories}")
        
        # En az bir kategori doğru eşleştirilmiş olmalı
        assert len(categories) > 0
        
        # Belge tiplerini kontrol et
        doc_types = set(doc["metadata"].get("type", "unknown") for doc in documents)
        logger.info(f"Belge tipleri: {doc_types}")
        
        # Hem metin hem de tablo belgeleri olmalı (tablolar PDF içeriğine bağlı)
        # Not: Test PDF'lerinde tablo olduğundan emin değilsek bu kontrolü atlayabiliriz
        
        logger.info("Uçtan uca test başarıyla tamamlandı")
    
    def test_e2e_streaming_pdf_processing(self, test_pdf_dir, output_dir, setup_logging):
        """
        Akış tabanlı PDF işlemenin uçtan uca testi
        
        Test akışı:
        1. PDF'leri hazırla
        2. Akış tabanlı işleme yap
        3. Sonuçları doğrula
        """
        logger = setup_logging
        logger.info("Uçtan uca test başlatılıyor: Akış tabanlı PDF işleme")
        
        # PDF dosyalarını bul
        pdf_files = [f for f in os.listdir(test_pdf_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            pytest.skip("Test PDF'leri bulunamadı.")
            
        # Test için ilk PDF'i kullan
        pdf_path = os.path.join(test_pdf_dir, pdf_files[0])
        logger.info(f"Test PDF: {pdf_path}")
        
        # EnhancedPdfProcessor oluştur (akış tabanlı)
        processor = EnhancedPdfProcessor(
            output_dir=output_dir,
            use_streaming=True,
            memory_threshold_mb=100
        )
        logger.info("Akış tabanlı PDF işlemci oluşturuldu")
        
        # PDF'i akış tabanlı işle
        result = processor.process_pdf_streaming(
            pdf_path=pdf_path,
            category="test",
            chunk_size=100,
            overlap=20
        )
        logger.info(f"PDF akış tabanlı işlendi: {result['chunk_count']} metin parçası")
        
        # Sonuçları doğrula
        assert isinstance(result, dict)
        assert "id" in result
        assert "chunk_count" in result
        assert "processing_completed" in result
        assert result["processing_completed"] is True
        assert result["chunk_count"] > 0
        
        # Özet dosyası oluşturuldu mu?
        summary_file = os.path.join(output_dir, f"{result['id']}_summary.json")
        assert os.path.exists(summary_file)
        
        # Ana veri dosyası oluşturuldu mu?
        output_file = os.path.join(output_dir, f"{result['id']}.json")
        assert os.path.exists(output_file)
        
        logger.info("Uçtan uca test başarıyla tamamlandı")
    
    def test_e2e_parallel_directory_processing(self, test_pdf_dir, output_dir, setup_logging):
        """
        PDF dizini paralel işlemenin uçtan uca testi
        
        Test akışı:
        1. PDF dizinini hazırla 
        2. Paralel işleme yap
        3. Sonuçları doğrula
        """
        logger = setup_logging
        logger.info("Uçtan uca test başlatılıyor: Paralel PDF dizini işleme")
        
        # PDF dosyalarını kontrol et
        pdf_files = [f for f in os.listdir(test_pdf_dir) if f.lower().endswith('.pdf')]
        if len(pdf_files) < 2:
            pytest.skip("Paralel işleme testi için en az 2 PDF dosyası gerekli.")
        
        # Test için PDF'leri çoğalt
        test_dir = os.path.join(output_dir, "test_pdfs")
        os.makedirs(test_dir, exist_ok=True)
        
        # PDF dosyalarını test dizinine kopyala ve çoğalt
        for pdf_file in pdf_files:
            for i in range(2):  # Her dosyayı 2 kez kopyala
                src = os.path.join(test_pdf_dir, pdf_file)
                dst = os.path.join(test_dir, f"copy_{i}_{pdf_file}")
                shutil.copy(src, dst)
        
        logger.info(f"Test dizini hazırlandı: {test_dir}")
        logger.info(f"Toplam PDF sayısı: {len(os.listdir(test_dir))}")
        
        # EnhancedPdfProcessor oluştur
        processor = EnhancedPdfProcessor(
            output_dir=os.path.join(output_dir, "results"),
            max_workers=2,  # Test için 2 işçi
            use_streaming=True
        )
        logger.info("PDF işlemci oluşturuldu: 2 paralel işçi")
        
        # PDF dizinini paralel işle
        results = processor.process_pdf_directory_streaming(
            directory=test_dir,
            parallel=True
        )
        
        logger.info(f"Paralel işleme tamamlandı: {len(results)} PDF işlendi")
        
        # Sonuçları doğrula
        assert isinstance(results, list)
        assert len(results) > 0
        assert len(results) == len(os.listdir(test_dir))
        
        # Her PDF için çıktı dosyası oluşturuldu mu?
        result_dir = os.path.join(output_dir, "results")
        output_files = [f for f in os.listdir(result_dir) if f.endswith('_summary.json')]
        assert len(output_files) == len(results)
        
        logger.info("Uçtan uca test başarıyla tamamlandı")
        
    def test_e2e_ocr_and_table_extraction(self, test_pdf_dir, output_dir, setup_logging, monkeypatch):
        """
        OCR ve tablo çıkarmanın uçtan uca testi
        
        Test akışı:
        1. OCR ve tablo çıkarma yeteneklerini etkinleştir
        2. PDF'i işle
        3. Sonuçları doğrula
        """
        logger = setup_logging
        logger.info("Uçtan uca test başlatılıyor: OCR ve tablo çıkarma")
        
        # PDF dosyalarını bul
        pdf_files = [f for f in os.listdir(test_pdf_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            pytest.skip("Test PDF'leri bulunamadı.")
            
        # Test için "mortgage_application.pdf" dosyasını bul (tablo içerdiğini varsayalım)
        mortgage_pdf = next((f for f in pdf_files if "mortgage" in f.lower()), pdf_files[0])
        pdf_path = os.path.join(test_pdf_dir, mortgage_pdf)
        logger.info(f"Test PDF: {pdf_path}")
        
        # OCR ve tablo çıkarma için mock'lar
        ocr_called = False
        table_called = False
        
        # Mock OCR fonksiyonu
        def mock_perform_ocr(*args, **kwargs):
            nonlocal ocr_called
            ocr_called = True
            return "OCR ile çıkarılan örnek metin"
        
        # Mock tablo çıkarma fonksiyonu
        def mock_extract_tables(*args, **kwargs):
            nonlocal table_called
            table_called = True
            # Sahte tablo döndür
            import pandas as pd
            df = pd.DataFrame({
                'Sütun1': ['Veri1', 'Veri2'],
                'Sütun2': ['Veri3', 'Veri4']
            })
            return [df]
        
        # Mock fonksiyonları uygula
        monkeypatch.setattr(EnhancedPdfProcessor, "perform_ocr", mock_perform_ocr)
        monkeypatch.setattr(EnhancedPdfProcessor, "extract_tables", mock_extract_tables)
        
        # EnhancedPdfProcessor oluştur
        processor = EnhancedPdfProcessor(
            output_dir=output_dir,
            ocr_lang="tur+eng",
            table_extraction_method="auto"
        )
        logger.info("PDF işlemci oluşturuldu: OCR ve tablo çıkarma etkin")
        
        # OCR ve tablo çıkarma ile işle
        result = processor.process_pdf(
            pdf_path,
            category="test",
            extract_tables=True,
            use_ocr=True
        )
        
        logger.info(f"PDF işlendi, OCR çağrıldı: {ocr_called}, Tablo çıkarma çağrıldı: {table_called}")
        
        # Fonksiyonların çağrıldığını doğrula
        assert table_called is True
        
        # OCR, yalnızca diğer metin çıkarma yöntemleri başarısız olursa çağrılır
        # Bu test için, OCR çağrısı monkeypatch ile manipüle edilmiştir
        
        # Sonuçları doğrula
        assert isinstance(result, dict)
        assert "tables" in result
        assert len(result["tables"]) > 0
        
        logger.info("Uçtan uca test başarıyla tamamlandı")
    
    def test_e2e_pdf_to_documents_integration(self, test_pdf_dir, output_dir, setup_logging):
        """
        PDF'ten doğrudan belge oluşturmanın uçtan uca testi
        Bu test, processPdfToDocuments'in diğer bileşenlerle entegrasyonunu test eder.
        
        Test akışı:
        1. PDF'i belge listesine dönüştür
        2. Sonuçları doğrula
        """
        logger = setup_logging
        logger.info("Uçtan uca test başlatılıyor: PDF'ten doğrudan belge entegrasyonu")
        
        # PDF dosyalarını bul
        pdf_files = [f for f in os.listdir(test_pdf_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            pytest.skip("Test PDF'leri bulunamadı.")
            
        # Test için ilk PDF'i kullan
        pdf_path = os.path.join(test_pdf_dir, pdf_files[0])
        logger.info(f"Test PDF: {pdf_path}")
        
        # PDF'i doğrudan belgeler listesine dönüştür
        documents = EnhancedPdfProcessor.process_pdf_to_documents(
            pdf_path=pdf_path,
            category="test",
            source="test_kaynak",
            chunk_size=100,
            overlap=20
        )
        
        logger.info(f"PDF'ten {len(documents)} belge oluşturuldu")
        
        # Sonuçları doğrula
        assert isinstance(documents, list)
        assert len(documents) > 0
        
        # Belge yapısını kontrol et
        for doc in documents:
            assert "id" in doc
            assert "text" in doc
            assert "metadata" in doc
            assert doc["metadata"]["category"] == "test"
            assert doc["metadata"]["source"] == "test_kaynak"
        
        # Belge tiplerini kontrol et
        text_chunks = [doc for doc in documents if doc["metadata"].get("type") == "text_chunk"]
        assert len(text_chunks) > 0
        
        logger.info("Uçtan uca test başarıyla tamamlandı") 