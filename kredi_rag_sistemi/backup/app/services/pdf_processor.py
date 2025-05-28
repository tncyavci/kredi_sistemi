import os
import logging
from typing import List, Dict, Optional, Any
from utils.preprocessing import EnhancedPdfProcessor
from pathlib import Path

logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Gelişmiş PDF işleme özelliklerini kullanarak PDF dosyalarını işleyen servis sınıfı.
    Bu sınıf, EnhancedPdfProcessor sınıfı üzerine bir servis adaptörü görevi görür.
    """
    def __init__(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        use_ocr: bool = True,
        extract_tables: bool = True
    ):
        """
        PDF işleme servisinin başlatıcısı.
        
        Args:
            input_dir: İşlenecek PDF dosyalarının bulunduğu dizin
            output_dir: İşlenmiş verilerin kaydedileceği dizin (belirtilmezse "data/processed")
            use_ocr: OCR kullanılsın mı
            extract_tables: Tablolar çıkarılsın mı
        """
        self.input_dir = input_dir
        self.output_dir = output_dir or os.path.join(os.getcwd(), "data/processed")
        self.use_ocr = use_ocr
        self.extract_tables = extract_tables
        
        # Gelişmiş PDF işleme sınıfını başlat
        self.processor = EnhancedPdfProcessor(
            output_dir=self.output_dir,
            ocr_lang="tur+eng",  # Türkçe ve İngilizce
            table_extraction_method="auto"  # Otomatik tablo çıkarma stratejisi
        )
        
        # Dizinlerin var olduğundan emin ol
        Path(self.input_dir).mkdir(exist_ok=True, parents=True)
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        
        # Kategori eşleştirmeleri
        self.category_mapping = {
            "mali": "mali_tablo",
            "faaliyet": "faaliyet_raporu",
            "fiyat": "fiyat_listesi",
            "finansal": "finansal_bilgiler",
            "kap": "finansal_bilgiler",
            "pegasus": "finansal_bilgiler",
            "adel": "mali_tablo",
            "kuzeykablo": "fiyat_listesi"
        }
    
    def process_pdf(self, pdf_path: str, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Bir PDF dosyasını işler.
        
        Args:
            pdf_path: PDF dosya yolu
            category: Belge kategorisi (belirtilmezse dosya adından çıkarılır)
            
        Returns:
            İşlenmiş PDF verileri
        """
        # Kategoriyi belirle
        if not category:
            filename = os.path.basename(pdf_path).lower()
            category = "genel"
            for pattern, cat in self.category_mapping.items():
                if pattern.lower() in filename:
                    category = cat
                    break
        
        # PDF'i işle
        return self.processor.process_pdf(
            pdf_path,
            category=category,
            extract_tables=self.extract_tables,
            use_ocr=self.use_ocr
        )
    
    def process_directory(self) -> List[Dict[str, Any]]:
        """
        Girdi dizinindeki tüm PDF dosyalarını işler.
        
        Returns:
            İşlenmiş PDF verilerinin listesi
        """
        return self.processor.process_pdf_directory(
            self.input_dir,
            category_mapping=self.category_mapping,
            extract_tables=self.extract_tables,
            use_ocr=self.use_ocr
        )
    
    def process_to_documents(self) -> List[Dict[str, Any]]:
        """
        Girdi dizinindeki tüm PDF dosyalarını belge formatına dönüştürür.
        
        Returns:
            İşlenmiş belgelerin listesi
        """
        return self.processor.process_pdf_directory_to_documents(
            self.input_dir,
            category_mapping=self.category_mapping,
            extract_tables=self.extract_tables,
            use_ocr=self.use_ocr
        )
    
    def extract_text_from_pdf(self, pdf_path: str, use_ocr: bool = True) -> str:
        """
        PDF dosyasından metin çıkarır.
        
        Args:
            pdf_path: PDF dosya yolu
            use_ocr: OCR kullanılsın mı
            
        Returns:
            PDF'ten çıkarılan metin
        """
        # Önce PyMuPDF ile dene
        text = self.processor.extract_text_with_pymupdf(pdf_path)
        
        # Metin çıkartılamadıysa pdfplumber dene
        if not text.strip():
            text = self.processor.extract_text_with_pdfplumber(pdf_path)
            
        # Hala metin yoksa ve OCR etkinse OCR dene
        if not text.strip() and use_ocr:
            text = self.processor.perform_ocr(pdf_path)
        
        return text
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        PDF dosyasından tabloları çıkarır.
        
        Args:
            pdf_path: PDF dosya yolu
            
        Returns:
            Çıkarılan tabloların JSON formatında listesi
        """
        tables = self.processor.extract_tables(pdf_path)
        return self.processor.tables_to_json(tables)
    
    def process_pdfs(self) -> List[Dict]:
        """
        Eski API uyumluluğu için tüm PDF'leri işler ve belge formatına dönüştürür.
        
        Returns:
            İşlenmiş belgelerin listesi
        """
        return self.process_to_documents()

def main():
    # PDF işleyici oluştur
    processor = PDFProcessor("data")
    
    # Tüm PDF'leri işle
    results = processor.process_directory()
    
    # Sonuçları kaydet
    output_file = processor.save_results(results)
    
    print(f"\nİşlenen PDF sayısı: {len(results)}")
    print(f"Sonuçlar şu dosyaya kaydedildi: {output_file}")
    
    # Her PDF için özet bilgileri göster
    print("\nPDF Özeti:")
    for doc in results:
        print(f"\nDosya: {doc['filename']}")
        print(f"Toplam Sayfa: {doc['total_pages']}")
        print(f"Toplam Kelime: {doc['total_words']}")

if __name__ == "__main__":
    main() 