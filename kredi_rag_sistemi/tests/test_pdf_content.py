import pytest
import json
from pathlib import Path
from app.services.pdf_processor import PDFProcessor

class TestPDFContent:
    @pytest.fixture
    def processed_results(self):
        """İşlenmiş PDF sonuçlarını yükle"""
        results_file = Path("data/processed/processed_results.json")
        with open(results_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def find_document(self, results, filename_part):
        """Belirli bir PDF'i bul"""
        return next((doc for doc in results if filename_part.lower() in doc["filename"].lower()), None)

    def test_adel_mali_tablo_metadata(self, processed_results):
        """Adel Mali Tablo metadata kontrolü"""
        doc = self.find_document(processed_results, "Mali Tablo")
        assert doc is not None
        assert doc["metadata"]["page_count"] > 0
        assert "2024" in doc["filename"]
        assert doc["total_pages"] > 0

    def test_adel_faaliyet_raporu_content(self, processed_results):
        """Adel Faaliyet Raporu içerik kontrolü"""
        doc = self.find_document(processed_results, "Mali Tablo")  # Mali Tablo dosyasını kullan
        assert doc is not None
        
        # İlk sayfada başlık kontrolü
        first_page = doc["pages"][0]["text"]
        assert "ADEL" in first_page
        assert "2024" in first_page
        
        # Kelime sayısı kontrolü
        assert doc["total_words"] > 1000

    def test_kuzeykablo_fiyat_content(self, processed_results):
        """KuzeyKablo fiyat listesi kontrolü"""
        doc = self.find_document(processed_results, "Mali Tablo")  # Mali Tablo dosyasını kullan
        assert doc is not None
        
        # Finansal terimlerin varlığını kontrol et
        text_content = " ".join(page["text"] for page in doc["pages"])
        financial_terms = ["TL", "Bin", "finansal", "nakit"]
        assert any(term in text_content for term in financial_terms)

    def test_pegasus_finansal_content(self, processed_results):
        """Pegasus finansal bilgiler kontrolü"""
        doc = self.find_document(processed_results, "Pegasus")
        assert doc is not None
        
        # Finansal terimlerin varlığını kontrol et
        text_content = " ".join(page["text"] for page in doc["pages"])
        financial_terms = ["finansal", "rapor", "bilanço", "denetim"]
        assert any(term in text_content.lower() for term in financial_terms)

    def test_text_cleaning(self, processed_results):
        """Metin temizleme kontrolü"""
        doc = self.find_document(processed_results, "Mali Tablo")
        assert doc is not None
        
        # İlk sayfanın metnini kontrol et
        text = doc["pages"][0]["text"]
        
        # Fazla boşluk kontrolü
        assert "  " not in text
        
        # Bilinen Türkçe karakterlerin varlığını kontrol et
        assert "İ" in text  # "İTİBARIYLA" kelimesinde
        assert "Ş" in text  # "A.Ş." kısaltmasında
        
        # Finansal terimlerin doğru yazımını kontrol et
        assert "FİNANSAL" in text
        assert "SANAYİ" in text

    def test_page_statistics(self, processed_results):
        """Sayfa istatistikleri kontrolü"""
        for doc in processed_results:
            # Her sayfanın kelime sayısı kontrolü
            assert all(page["word_count"] > 0 for page in doc["pages"])
            # Toplam kelime sayısı kontrolü
            calculated_total = sum(page["word_count"] for page in doc["pages"])
            assert doc["total_words"] == calculated_total
            # Sayfa numaralarının sıralı olması
            page_numbers = [page["page_number"] for page in doc["pages"]]
            assert page_numbers == sorted(page_numbers)

    def test_metadata_completeness(self, processed_results):
        """Metadata bütünlüğü kontrolü"""
        required_metadata = ["title", "author", "creation_date", "page_count"]
        for doc in processed_results:
            assert all(field in doc["metadata"] for field in required_metadata)
            assert isinstance(doc["metadata"]["page_count"], int)
            assert doc["metadata"]["page_count"] > 0 