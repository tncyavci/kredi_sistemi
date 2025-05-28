import os
import uuid
from typing import List, Dict, Any, Optional
import logging
from PyPDF2 import PdfReader
import gc
from pathlib import Path

logger = logging.getLogger(__name__)

class PdfProcessor:
    """
    PDF dosyalarını işleyen ve metin çıkaran yardımcı sınıf.
    RAG sistemi için finansal PDF dokümanlarından metin çıkarma işlemlerini yapar.
    """
    
    @staticmethod
    def _validate_pdf_file(pdf_path: str) -> bool:
        """
        PDF dosyasının geçerliliğini kontrol eder.
        
        Args:
            pdf_path: PDF dosyasının yolu
            
        Returns:
            Dosya geçerliyse True, değilse False
        """
        try:
            # Dosya var mı kontrol et
            if not os.path.exists(pdf_path):
                logger.error(f"Dosya bulunamadı: {pdf_path}")
                return False
                
            # Dosya boyutu kontrolü (max 50MB)
            file_size = os.path.getsize(pdf_path)
            if file_size > 50 * 1024 * 1024:  # 50MB
                logger.error(f"Dosya boyutu çok büyük: {pdf_path} ({file_size/1024/1024:.2f}MB)")
                return False
                
            # Dosya uzantısı kontrolü
            if not pdf_path.lower().endswith('.pdf'):
                logger.error(f"Geçersiz dosya uzantısı: {pdf_path}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Dosya doğrulama hatası ({pdf_path}): {str(e)}")
            return False
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str, max_pages: Optional[int] = None) -> str:
        """
        PDF dosyasından tam metni çıkarır.
        
        Args:
            pdf_path: PDF dosyasının yolu
            max_pages: İşlenecek maksimum sayfa sayısı (bellek sorunlarını önlemek için)
            
        Returns:
            PDF'ten çıkarılan metin
        """
        # PDF dosyasını doğrula
        if not PdfProcessor._validate_pdf_file(pdf_path):
            return ""
            
        reader = None
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            # Sayfa sayısını sınırla
            if max_pages is None:
                max_pages = len(reader.pages)
            else:
                max_pages = min(max_pages, len(reader.pages))
                
            logger.info(f"PDF işleniyor: {pdf_path} - {max_pages} sayfa")
            
            for page_num in range(max_pages):
                try:
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Sayfa {page_num+1} ---\n{page_text}"
                    
                    # Her 10 sayfada bir belleği temizle
                    if page_num % 10 == 0 and page_num > 0:
                        gc.collect()
                        
                except Exception as e:
                    logger.error(f"Sayfa işleme hatası ({pdf_path}, sayfa {page_num+1}): {str(e)}")
                    continue
            
            # Son belleği temizle
            gc.collect()
            return text
            
        except Exception as e:
            logger.error(f"PDF işlenirken hata oluştu: {pdf_path} - {str(e)}")
            return ""
            
        finally:
            # Reader'ı temizle
            if reader:
                del reader
            gc.collect()
            
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Uzun metni belli bir boyut ve örtüşme oranıyla parçalara böler.
        
        Args:
            text: Bölünecek metin
            chunk_size: Her bir parçanın maksimum karakter sayısı
            overlap: Ardışık parçalar arasındaki örtüşme miktarı
            
        Returns:
            Metin parçalarının listesi
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # Eğer cümle ortasında kesiliyorsa, bir sonraki noktaya kadar ilerle
            if end < text_length:
                # Noktadan sonraki ilk boşluğu bul
                next_period = text.find('. ', end)
                if next_period != -1 and next_period - end < 100:  # Makul bir aralıkta nokta varsa
                    end = next_period + 2  # Noktayı ve boşluğu dahil et
            
            chunks.append(text[start:end])
            start = end - overlap  # Örtüşme miktarı kadar geri git
            
            # Her 20 parçada bir belleği temizle
            if len(chunks) % 20 == 0:
                gc.collect()
            
        return chunks
        
    @staticmethod
    def process_pdf_to_documents(
        pdf_path: str, 
        category: str = "finansal_rapor",
        source: Optional[str] = None,
        chunk_size: int = 800,
        overlap: int = 150,
        max_pages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        PDF dosyasını işleyerek RAG sistemi için belge listesi oluşturur.
        
        Args:
            pdf_path: PDF dosyasının yolu
            category: Belgenin kategorisi
            source: Belgenin kaynağı (belirtilmezse dosya adı kullanılır)
            chunk_size: Her bir metin parçasının maksimum karakter sayısı
            overlap: Ardışık parçalar arasındaki örtüşme miktarı
            max_pages: İşlenecek maksimum sayfa sayısı
            
        Returns:
            RAG sistemi için hazır belge listesi
        """
        # Dosya adını al (source belirtilmemişse)
        if not source:
            source = os.path.basename(pdf_path)
        
        # Metni çıkar
        full_text = PdfProcessor.extract_text_from_pdf(pdf_path, max_pages=max_pages)
        if not full_text:
            logger.warning(f"PDF'ten metin çıkarılamadı: {pdf_path}")
            return []
            
        # Metni parçala
        chunks = PdfProcessor.chunk_text(full_text, chunk_size, overlap)
        
        # Her bir parça için belge oluştur
        documents = []
        for i, chunk in enumerate(chunks):
            doc_id = f"{uuid.uuid4()}"
            documents.append({
                "id": doc_id,
                "text": chunk,
                "metadata": {
                    "source": source,
                    "category": category,
                    "page_range": f"Parça {i+1}/{len(chunks)}",
                    "file_path": pdf_path
                }
            })
            
        logger.info(f"PDF işlendi: {pdf_path}, {len(documents)} belge oluşturuldu")
        return documents
        
    @staticmethod
    def process_pdf_directory(
        dir_path: str,
        category_mapping: Optional[Dict[str, str]] = None,
        chunk_size: int = 800,
        overlap: int = 150,
        max_pages_per_pdf: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Bir dizindeki tüm PDF dosyalarını işler.
        
        Args:
            dir_path: PDF dosyalarının bulunduğu dizin
            category_mapping: Dosya adı kalıplarına göre kategori eşleştirmeleri
            chunk_size: Her bir metin parçasının maksimum karakter sayısı
            overlap: Ardışık parçalar arasındaki örtüşme miktarı
            max_pages_per_pdf: Her PDF için işlenecek maksimum sayfa sayısı
            
        Returns:
            Tüm PDF'lerden oluşturulmuş belge listesi
        """
        if not os.path.exists(dir_path):
            logger.error(f"Dizin bulunamadı: {dir_path}")
            return []
            
        all_documents = []
        pdf_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.pdf')]
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(dir_path, pdf_file)
            
            # Dosyaya uygun kategoriyi belirle
            category = "finansal_rapor"  # Varsayılan kategori
            if category_mapping:
                for pattern, cat in category_mapping.items():
                    if pattern.lower() in pdf_file.lower():
                        category = cat
                        break
                        
            # PDF'i işle ve belgeleri listeye ekle
            documents = PdfProcessor.process_pdf_to_documents(
                pdf_path, 
                category=category,
                source=pdf_file,
                chunk_size=chunk_size,
                overlap=overlap,
                max_pages=max_pages_per_pdf
            )
            all_documents.extend(documents)
            
            # Her PDF'den sonra belleği temizle
            gc.collect()
            
        logger.info(f"{len(pdf_files)} PDF dosyası işlendi, toplam {len(all_documents)} belge oluşturuldu")
        return all_documents 