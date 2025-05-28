import os
import logging
from typing import List, Dict
from utils.preprocessing.enhanced_pdf_processor import EnhancedPdfProcessor

logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Gelişmiş PDF işleme özelliklerini kullanarak PDF dosyalarını işleyen sınıf.
    Bu sınıf, EnhancedPdfProcessor sınıfı üzerine bir adaptör görevi görür.
    """
    def __init__(self, pdf_dir: str):
        self.pdf_dir = pdf_dir
        self.processor = EnhancedPdfProcessor(output_dir=os.path.join(pdf_dir, "../data/processed"))

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF dosyasından metin çıkarır"""
        try:
            # Önce PyMuPDF ile dene
            text = self.processor.extract_text_with_pymupdf(pdf_path)
            
            # Eğer metin bulunamazsa pdfplumber dene
            if not text.strip():
                text = self.processor.extract_text_with_pdfplumber(pdf_path)
                
            # Hala metin yoksa OCR dene
            if not text.strip():
                text = self.processor.perform_ocr(pdf_path)
                
            return text
        except Exception as e:
            logger.error(f"PDF işlenirken hata oluştu: {pdf_path}, Hata: {str(e)}")
            raise

    def process_pdfs(self) -> List[Dict]:
        """Tüm PDF'leri işler ve belge formatına dönüştürür"""
        try:
            return self.processor.process_pdf_directory_to_documents(self.pdf_dir, category_mapping={
                "mali": "mali_tablo",
                "faaliyet": "faaliyet_raporu",
                "fiyat": "fiyat_listesi",
                "finansal": "finansal_bilgiler",
                "kap": "finansal_bilgiler",
                "pegasus": "finansal_bilgiler"
            })
        except Exception as e:
            logger.error(f"PDF dizini işlenirken hata oluştu: {str(e)}")
            return [] 