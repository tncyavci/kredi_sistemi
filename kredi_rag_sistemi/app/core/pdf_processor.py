import os
import logging
from typing import List, Dict
import PyPDF2
import pdfplumber

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, pdf_dir: str):
        self.pdf_dir = pdf_dir

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF dosyasından metin çıkarır"""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"PDF işlenirken hata oluştu: {pdf_path}, Hata: {str(e)}")
            raise
        return text

    def process_pdfs(self) -> List[Dict]:
        """Tüm PDF'leri işler ve belge formatına dönüştürür"""
        documents = []
        for filename in os.listdir(self.pdf_dir):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_dir, filename)
                try:
                    text = self.extract_text_from_pdf(pdf_path)
                    document = {
                        "id": f"pdf_{os.path.splitext(filename)[0]}",
                        "text": text,
                        "metadata": {
                            "category": "finansal_rapor",
                            "kaynak": filename,
                            "type": "pdf"
                        }
                    }
                    documents.append(document)
                    logger.info(f"PDF işlendi: {filename}")
                except Exception as e:
                    logger.error(f"PDF işlenemedi: {filename}, Hata: {str(e)}")
                    continue
        return documents 