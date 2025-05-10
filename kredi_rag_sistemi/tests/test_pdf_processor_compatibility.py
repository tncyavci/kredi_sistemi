import os
import pytest
import tempfile
import json
from pathlib import Path

from utils.preprocessing.pdf_processor import PdfProcessor
from utils.preprocessing.enhanced_pdf_processor import EnhancedPdfProcessor

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

class TestPdfProcessorCompatibility:
    """
    PdfProcessor ve EnhancedPdfProcessor arasındaki uyumluluk testleri.
    Bu testler, eski kodla yeni kodun sorunsuz çalışabildiğini doğrular.
    """
    
    def test_pdf_processor_output_compatibility(self, test_pdf_dir, output_dir):
        """
        Her iki işlemci tarafından üretilen çıktıların uyumlu olduğunu doğrular.
        """
        # PDF dosyalarını bul
        pdf_files = [f for f in os.listdir(test_pdf_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            pytest.skip("Test PDF'leri bulunamadı.")
            
        # Test için ilk PDF'i kullan
        pdf_path = os.path.join(test_pdf_dir, pdf_files[0])
        
        # Eski PdfProcessor ile çıktı oluştur
        old_processor_output_dir = os.path.join(output_dir, "old_processor")
        os.makedirs(old_processor_output_dir, exist_ok=True)
        
        old_processor = PdfProcessor()
        old_result = old_processor.process_pdf_to_documents(
            pdf_path=pdf_path,
            category="test",
            source="test_source",
            chunk_size=100,
            overlap=20
        )
        
        # Yeni EnhancedPdfProcessor ile çıktı oluştur
        new_processor_output_dir = os.path.join(output_dir, "new_processor")
        os.makedirs(new_processor_output_dir, exist_ok=True)
        
        new_processor = EnhancedPdfProcessor(output_dir=new_processor_output_dir)
        new_result = new_processor.process_pdf_to_documents(
            pdf_path=pdf_path,
            category="test",
            source="test_source",
            chunk_size=100,
            overlap=20
        )
        
        # Sonuçları karşılaştır
        
        # Belge sayıları (yaklaşık olarak) aynı olmalı
        # Not: OCR ve tablo tanıma özellikleri nedeniyle tam olarak aynı olmayabilir
        assert abs(len(old_result) - len(new_result)) < 3, "Belge sayıları çok farklı"
        
        # Belge yapısı aynı olmalı
        if old_result and new_result:
            old_doc = old_result[0]
            new_doc = new_result[0]
            
            # Anahtar alanlar her iki sonuçta da mevcut olmalı
            assert "id" in old_doc and "id" in new_doc
            assert "text" in old_doc and "text" in new_doc
            assert "metadata" in old_doc and "metadata" in new_doc
            
            # Meta veri alanları
            assert "source" in old_doc["metadata"] and "source" in new_doc["metadata"]
            assert "category" in old_doc["metadata"] and "category" in new_doc["metadata"]
            
            # Değerler aynı olmalı
            assert old_doc["metadata"]["source"] == new_doc["metadata"]["source"]
            assert old_doc["metadata"]["category"] == new_doc["metadata"]["category"]
    
    def test_cross_processor_compatibility(self, test_pdf_dir, output_dir):
        """
        Bir işlemciden diğerine geçişin sorunsuz çalıştığını doğrular.
        """
        # PDF dosyalarını bul
        pdf_files = [f for f in os.listdir(test_pdf_dir) if f.lower().endswith('.pdf')]
        if len(pdf_files) < 2:
            pytest.skip("Bu test için en az 2 PDF dosyası gerekli.")
            
        # Test için iki PDF seç
        pdf_path1 = os.path.join(test_pdf_dir, pdf_files[0])
        pdf_path2 = os.path.join(test_pdf_dir, pdf_files[1])
        
        # İlk PDF'i eski işlemci ile, ikinci PDF'i yeni işlemci ile işle
        old_processor = PdfProcessor()
        new_processor = EnhancedPdfProcessor(output_dir=output_dir)
        
        # İlk PDF'i eski işlemci ile işle
        old_result = old_processor.process_pdf_to_documents(
            pdf_path=pdf_path1,
            category="test_old",
            source="test_source_old"
        )
        
        # İkinci PDF'i yeni işlemci ile işle
        new_result = new_processor.process_pdf_to_documents(
            pdf_path=pdf_path2,
            category="test_new",
            source="test_source_new"
        )
        
        # Her iki işlemciden gelen sonuçları birleştir
        combined_result = old_result + new_result
        
        # Birleştirilmiş sonucun tutarlı olduğunu doğrula
        assert len(combined_result) == len(old_result) + len(new_result)
        
        # Her sonuç kendi meta verilerini korumalı
        old_docs = [doc for doc in combined_result if doc["metadata"]["category"] == "test_old"]
        new_docs = [doc for doc in combined_result if doc["metadata"]["category"] == "test_new"]
        
        assert len(old_docs) == len(old_result)
        assert len(new_docs) == len(new_result)
    
    def test_processor_interchangeability(self, test_pdf_dir, output_dir):
        """
        İşlemcilerin birbirinin yerine kullanılabilirliğini test eder.
        """
        # PDF dosyasını bul
        pdf_files = [f for f in os.listdir(test_pdf_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            pytest.skip("Test PDF'leri bulunamadı.")
            
        pdf_path = os.path.join(test_pdf_dir, pdf_files[0])
        
        # EnhancedPdfProcessor'ı PdfProcessor'ın beklediği şekilde kullan
        # (örnek olarak process_pdf_to_documents statik metodu)
        
        # Eski tarzda çağrı
        old_style_result = PdfProcessor.process_pdf_to_documents(
            pdf_path=pdf_path,
            category="test",
            source="test_source"
        )
        
        # Yeni tarzda çağrı (aynı statik metodu)
        new_style_result = EnhancedPdfProcessor.process_pdf_to_documents(
            pdf_path=pdf_path,
            category="test",
            source="test_source"
        )
        
        # Sonuçlar aynı formatta olmalı
        assert isinstance(old_style_result, list)
        assert isinstance(new_style_result, list)
        
        # Her iki sonuç da belgeler içermeli
        assert len(old_style_result) > 0
        assert len(new_style_result) > 0
    
    def test_adapter_pattern_compatibility(self, test_pdf_dir, output_dir):
        """
        Bir adaptör deseni ile eski kodu yeni işlemciyle kullanabilmeyi test eder.
        Bu test, eski kodda değişiklik yapmadan yeni işlemciyi kullanmayı simüle eder.
        """
        # Adaptör sınıfı: Eski PdfProcessor arayüzünü korur, yeni EnhancedPdfProcessor'ı kullanır
        class PdfProcessorAdapter(PdfProcessor):
            def __init__(self, output_dir=None):
                super().__init__()
                self.enhanced_processor = EnhancedPdfProcessor(output_dir=output_dir)
            
            def process_pdf_to_documents(self, pdf_path, **kwargs):
                # Eski yöntemi çağırmak yerine yeni yöntemi kullan
                return self.enhanced_processor.process_pdf_to_documents(pdf_path, **kwargs)
        
        # PDF dosyasını bul
        pdf_files = [f for f in os.listdir(test_pdf_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            pytest.skip("Test PDF'leri bulunamadı.")
            
        pdf_path = os.path.join(test_pdf_dir, pdf_files[0])
        
        # Adaptör ile işle
        adapter = PdfProcessorAdapter(output_dir=output_dir)
        adapter_result = adapter.process_pdf_to_documents(
            pdf_path=pdf_path,
            category="test",
            source="test_source"
        )
        
        # Doğrudan yeni işlemci ile işle
        enhanced_processor = EnhancedPdfProcessor(output_dir=output_dir)
        enhanced_result = enhanced_processor.process_pdf_to_documents(
            pdf_path=pdf_path,
            category="test",
            source="test_source"
        )
        
        # Sonuçlar benzer olmalı
        assert len(adapter_result) == len(enhanced_result)
        
        # Her belge aynı yapıya sahip olmalı
        for i in range(min(len(adapter_result), len(enhanced_result))):
            assert "id" in adapter_result[i] and "id" in enhanced_result[i]
            assert "text" in adapter_result[i] and "text" in enhanced_result[i]
            assert "metadata" in adapter_result[i] and "metadata" in enhanced_result[i] 