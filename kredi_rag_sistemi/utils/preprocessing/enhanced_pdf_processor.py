"""
Gelişmiş PDF işleme modülü.
OCR, tablo tanıma ve yapılandırılmış veri çıkarma özellikleriyle donatılmış
PDF işleme işlevleri sağlar.
"""

import os
import re
import logging
import tempfile
import gc
import psutil
import threading
from typing import List, Dict, Any, Optional, Tuple, Union, Generator, Iterator
from pathlib import Path
import json
from functools import wraps
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

# PDF işleme kütüphaneleri
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
import camelot
import tabula
import pandas as pd
import numpy as np

# OCR için kütüphaneler
import pytesseract
from pdf2image import convert_from_path
import cv2

logger = logging.getLogger(__name__)

# Bellek izleme ve temizleme için yardımcı fonksiyonlar
def get_memory_usage() -> float:
    """Mevcut işlemin bellek kullanımını MB cinsinden döndürür"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    memory_mb = mem_info.rss / (1024 * 1024)
    return memory_mb

def memory_monitor(threshold_mb: float = 1000, check_interval: float = 5.0):
    """Bellek kullanımını izleyen dekoratör"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # İzleme için thread oluştur
            stop_event = threading.Event()
            
            def monitor_memory():
                while not stop_event.is_set():
                    mem_usage = get_memory_usage()
                    if mem_usage > threshold_mb:
                        logger.warning(f"Yüksek bellek kullanımı: {mem_usage:.2f} MB. GC çağrılıyor.")
                        gc.collect()
                    stop_event.wait(check_interval)
            
            # İzleme thread'ini başlat
            monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
            monitor_thread.start()
            
            try:
                # Asıl fonksiyonu çalıştır
                result = func(*args, **kwargs)
                return result
            finally:
                # İzleme thread'ini durdur
                stop_event.set()
                gc.collect()
        
        return wrapper
    return decorator

def force_gc_after(func):
    """Bir fonksiyondan sonra garbage collection'ı zorlayan dekoratör"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        gc.collect()
        return result
    return wrapper

class EnhancedPdfProcessor:
    """
    Gelişmiş PDF işleme yetenekleri sunan sınıf.
    Bu sınıf, OCR ve tablo tanıma dahil olmak üzere çeşitli PDF işleme işlevleri sağlar.
    """
    
    def __init__(
        self,
        output_dir: str = "./data/processed",
        ocr_lang: str = "tur+eng",  # OCR dil seçeneği (Türkçe + İngilizce)
        ocr_threshold: float = 0.7,  # OCR güven eşiği
        table_extraction_method: str = "auto",  # 'camelot', 'tabula' veya 'auto'
        memory_threshold_mb: float = 1000,  # Bellek temizleme eşiği (MB)
        use_streaming: bool = True,  # Akış tabanlı işleme kullanılsın mı
        max_workers: int = 4  # Paralel işleme için maksimum iş parçacığı sayısı
    ):
        """
        EnhancedPdfProcessor sınıfının başlatıcısı.
        
        Args:
            output_dir: İşlenmiş verilerin kaydedileceği dizin
            ocr_lang: OCR için dil seçeneği
            ocr_threshold: OCR sonuçları için güven eşiği
            table_extraction_method: Tablo çıkarma yöntemi ('camelot', 'tabula' veya 'auto')
            memory_threshold_mb: Bellek temizleme eşiği (MB)
            use_streaming: Akış tabanlı işleme kullanılsın mı
            max_workers: Paralel işleme için maksimum iş parçacığı sayısı
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.ocr_lang = ocr_lang
        self.ocr_threshold = ocr_threshold
        self.table_extraction_method = table_extraction_method
        self.memory_threshold_mb = memory_threshold_mb
        self.use_streaming = use_streaming
        self.max_workers = max_workers
        
        self.text_cleaners = [
            self._remove_extra_spaces,
            self._remove_special_characters,
            self._normalize_whitespace
        ]
        
        # Tesseract OCR'ın yüklü olduğunu kontrol et
        try:
            pytesseract.get_tesseract_version()
            self.ocr_available = True
            logger.info("Tesseract OCR bulundu ve kullanılabilir.")
        except Exception as e:
            self.ocr_available = False
            logger.warning(f"Tesseract OCR kullanılamıyor: {str(e)}")
    
    def _check_memory_usage(self) -> bool:
        """
        Bellek kullanımını kontrol eder ve eşik değeri aşılmışsa GC çağırır.
        
        Returns:
            Bellek temizleme işlemi gerçekleştiyse True, yoksa False
        """
        memory_mb = get_memory_usage()
        if memory_mb > self.memory_threshold_mb:
            logger.warning(f"Bellek kullanımı eşiği aşıldı: {memory_mb:.2f} MB. Bellek temizleniyor...")
            gc.collect()
            return True
        return False
    
    @staticmethod
    def _validate_pdf_file(file_path: str) -> bool:
        """
        PDF dosyasının geçerli olduğunu doğrular.
        
        Args:
            file_path: PDF dosya yolu
            
        Returns:
            Dosya geçerli bir PDF ise True, değilse False
        """
        if not os.path.exists(file_path):
            logger.warning(f"Dosya bulunamadı: {file_path}")
            return False
        
        if not file_path.lower().endswith('.pdf'):
            logger.warning(f"Geçersiz dosya uzantısı: {file_path}")
            return False
        
        try:
            # Dosya boyutunu kontrol et
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > 100:  # 100MB üst sınır
                logger.warning(f"PDF dosyası çok büyük: {file_path} ({file_size_mb:.2f} MB)")
                return False
            
            # PDF'in açılabilirliğini kontrol et
            with open(file_path, 'rb') as f:
                PyPDF2.PdfReader(f)
            
            return True
        except Exception as e:
            logger.error(f"PDF doğrulama hatası: {file_path}, {str(e)}")
            return False
    
    def _remove_extra_spaces(self, text: str) -> str:
        """Fazla boşlukları temizler"""
        return re.sub(r'\s+', ' ', text).strip()
    
    def _remove_special_characters(self, text: str) -> str:
        """Özel karakterleri temizler"""
        # Türkçe karakterleri koru - basitleştirilmiş regex
        pattern = r'[^\w\sçÇğĞıİöÖşŞüÜ.,;:!?()%&$#@+\-/*="\']'
        return re.sub(pattern, '', text)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Boşlukları normalize eder"""
        return re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()
    
    def clean_text(self, text: str) -> str:
        """
        Metni temizler ve normalize eder.
        
        Args:
            text: Temizlenecek metin
            
        Returns:
            Temizlenmiş metin
        """
        for cleaner in self.text_cleaners:
            text = cleaner(text)
        return text
    
    def extract_text_with_pdfplumber(self, pdf_path: str, max_pages: Optional[int] = None) -> str:
        """
        PDF'ten pdfplumber kullanarak metin çıkarır.
        
        Args:
            pdf_path: PDF dosya yolu
            max_pages: İşlenecek maksimum sayfa sayısı (None=tümü)
            
        Returns:
            PDF'ten çıkarılan metin
        """
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages = pdf.pages[:max_pages] if max_pages else pdf.pages
                for page in pages:
                    text += page.extract_text() or ""
                    text += "\n\n"
        except Exception as e:
            logger.error(f"pdfplumber ile metin çıkarırken hata: {str(e)}")
        
        return text
    
    def extract_text_with_pymupdf(self, pdf_path: str, max_pages: Optional[int] = None) -> str:
        """
        PDF'ten PyMuPDF (fitz) kullanarak metin çıkarır.
        
        Args:
            pdf_path: PDF dosya yolu
            max_pages: İşlenecek maksimum sayfa sayısı (None=tümü)
            
        Returns:
            PDF'ten çıkarılan metin
        """
        text = ""
        try:
            doc = fitz.open(pdf_path)
            page_count = doc.page_count
            pages_to_process = min(page_count, max_pages or page_count)
            
            for page_num in range(pages_to_process):
                page = doc.load_page(page_num)
                text += page.get_text()
                text += "\n\n"
            
            doc.close()
        except Exception as e:
            logger.error(f"PyMuPDF ile metin çıkarırken hata: {str(e)}")
        
        return text
    
    def perform_ocr(self, pdf_path: str, page_numbers: Optional[List[int]] = None) -> str:
        """
        PDF üzerinde OCR gerçekleştirir.
        
        Args:
            pdf_path: PDF dosya yolu
            page_numbers: OCR uygulanacak sayfa numaraları (None=tümü)
            
        Returns:
            OCR ile çıkarılan metin
        """
        if not self.ocr_available:
            logger.warning("Tesseract OCR kullanılamıyor. OCR atlanıyor.")
            return ""
        
        text = ""
        try:
            # PDF'i görüntülere dönüştür
            images = convert_from_path(pdf_path)
            
            # Belirli sayfaları işle
            if page_numbers:
                selected_images = [images[i] for i in page_numbers if i < len(images)]
            else:
                selected_images = images
            
            # Her görüntü için OCR uygula
            for i, image in enumerate(selected_images):
                # Görüntüyü geçici olarak kaydet
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    temp_path = tmp.name
                
                image.save(temp_path, 'PNG')
                
                # OpenCV ile görüntüyü oku ve ön işleme yap
                img = cv2.imread(temp_path)
                
                # Gri tonlamaya dönüştür
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Gürültü azaltma
                gray = cv2.medianBlur(gray, 3)
                
                # Adaptif eşikleme
                thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                
                # OCR uygula
                page_text = pytesseract.image_to_string(thresh, lang=self.ocr_lang)
                text += page_text + "\n\n"
                
                # Geçici dosyayı sil
                os.unlink(temp_path)
                
            logger.info(f"OCR başarıyla uygulandı: {len(selected_images)} sayfa.")
        except Exception as e:
            logger.error(f"OCR uygulanırken hata oluştu: {str(e)}")
        
        return text
    
    def extract_tables_with_camelot(self, pdf_path: str, pages: str = "all") -> List[pd.DataFrame]:
        """
        PDF'ten camelot kullanarak tabloları çıkarır.
        
        Args:
            pdf_path: PDF dosya yolu
            pages: İşlenecek sayfalar ('all' veya '1,2,3-5' formatında)
            
        Returns:
            Çıkarılan tabloların DataFrame listesi
        """
        tables = []
        try:
            # Camelot ile tabloları çıkar
            extracted_tables = camelot.read_pdf(
                pdf_path,
                pages=pages,
                flavor='lattice',  # Çizgiler ve sınırlar olan tablolar için
                suppress_stdout=True
            )
            
            logger.info(f"Camelot ile {len(extracted_tables)} tablo bulundu.")
            
            # Kaliteli ve kullanılabilir tabloları filtrele
            for i, table in enumerate(extracted_tables):
                if table.accuracy > 80:  # %80'den yüksek doğruluk
                    tables.append(table.df)
        except Exception as e:
            logger.error(f"Camelot ile tablo çıkarırken hata: {str(e)}")
            
            # İkinci bir deneme: stream flavor
            try:
                extracted_tables = camelot.read_pdf(
                    pdf_path,
                    pages=pages,
                    flavor='stream',  # Çizgisiz tablolar için
                    suppress_stdout=True
                )
                
                for i, table in enumerate(extracted_tables):
                    if table.accuracy > 70:  # Stream modunda daha düşük eşik
                        tables.append(table.df)
            except Exception as e2:
                logger.error(f"Camelot stream ile tablo çıkarırken hata: {str(e2)}")
        
        return tables
    
    def extract_tables_with_tabula(self, pdf_path: str, pages: Union[str, List[int]] = "all") -> List[pd.DataFrame]:
        """
        PDF'ten tabula kullanarak tabloları çıkarır.
        
        Args:
            pdf_path: PDF dosya yolu
            pages: İşlenecek sayfalar ('all' veya sayfa numaraları listesi)
            
        Returns:
            Çıkarılan tabloların DataFrame listesi
        """
        try:
            # Tabula ile tabloları çıkar
            return tabula.read_pdf(
                pdf_path,
                pages=pages,
                multiple_tables=True
            )
        except Exception as e:
            logger.error(f"Tabula ile tablo çıkarırken hata: {str(e)}")
            return []
    
    def extract_tables(self, pdf_path: str, pages: str = "all") -> List[pd.DataFrame]:
        """
        En iyi yöntemi kullanarak PDF'ten tabloları çıkarır.
        
        Args:
            pdf_path: PDF dosya yolu
            pages: İşlenecek sayfalar
            
        Returns:
            Çıkarılan tabloların DataFrame listesi
        """
        tables = []
        
        if self.table_extraction_method == "camelot":
            tables = self.extract_tables_with_camelot(pdf_path, pages)
        elif self.table_extraction_method == "tabula":
            tables = self.extract_tables_with_tabula(pdf_path, pages)
        else:  # "auto" mod - her iki yöntemi de dene
            # Önce camelot dene
            tables = self.extract_tables_with_camelot(pdf_path, pages)
            
            # Eğer camelot tablo bulamadıysa tabula'yı dene
            if not tables:
                tables = self.extract_tables_with_tabula(pdf_path, pages)
        
        logger.info(f"{len(tables)} tablo başarıyla çıkarıldı.")
        return tables
    
    def tables_to_text(self, tables: List[pd.DataFrame]) -> str:
        """
        Tabloları metin formatına dönüştürür.
        
        Args:
            tables: DataFrame listesi
            
        Returns:
            Tabloların metin gösterimi
        """
        text = ""
        for i, table in enumerate(tables):
            text += f"Tablo {i+1}:\n"
            text += table.to_string(index=False) + "\n\n"
        return text
    
    def tables_to_json(self, tables: List[pd.DataFrame]) -> List[Dict]:
        """
        Tabloları JSON formatına dönüştürür.
        
        Args:
            tables: DataFrame listesi
            
        Returns:
            Tabloların JSON gösterimi
        """
        result = []
        for i, table in enumerate(tables):
            # NaN değerleri None ile değiştir
            table_dict = table.where(pd.notnull(table), None).to_dict(orient='records')
            result.append({
                "table_id": i+1,
                "headers": list(table.columns),
                "data": table_dict
            })
        return result
        
    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        PDF'in meta verilerini çıkarır.
        
        Args:
            pdf_path: PDF dosya yolu
            
        Returns:
            PDF meta verileri
        """
        metadata = {
            "filename": os.path.basename(pdf_path),
            "file_size_kb": os.path.getsize(pdf_path) / 1024,
            "created_at": None,
            "modified_at": None,
            "author": None,
            "title": None,
            "subject": None,
            "keywords": None,
            "total_pages": 0
        }
        
        try:
            # PyPDF2 ile meta verileri çıkar
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                metadata["total_pages"] = len(reader.pages)
                
                if reader.metadata:
                    info = reader.metadata
                    metadata["author"] = info.get('/Author')
                    metadata["title"] = info.get('/Title')
                    metadata["subject"] = info.get('/Subject')
                    metadata["keywords"] = info.get('/Keywords')
                    
                    # Tarihler
                    if info.get('/CreationDate'):
                        metadata["created_at"] = str(info.get('/CreationDate'))
                    if info.get('/ModDate'):
                        metadata["modified_at"] = str(info.get('/ModDate'))
                        
            # PyMuPDF ile daha fazla detay
            try:
                doc = fitz.open(pdf_path)
                # Sayfa boyutları
                first_page = doc[0]
                metadata["page_size"] = {
                    "width": first_page.rect.width,
                    "height": first_page.rect.height
                }
                doc.close()
            except Exception as e:
                logger.warning(f"PyMuPDF meta verisi çıkarılamadı: {str(e)}")
                
        except Exception as e:
            logger.error(f"Meta veri çıkarırken hata: {str(e)}")
        
        return metadata
        
    def process_pdf(
        self, 
        pdf_path: str, 
        category: str = "genel", 
        extract_tables: bool = True,
        use_ocr: bool = True,
        max_pages: Optional[int] = None,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> Dict[str, Any]:
        """
        Bir PDF dosyasını tam olarak işler.
        
        Args:
            pdf_path: İşlenecek PDF dosya yolu
            category: Belge kategorisi
            extract_tables: Tabloları çıkar
            use_ocr: OCR uygula
            max_pages: İşlenecek maksimum sayfa sayısı
            chunk_size: Metin parçalama uzunluğu
            overlap: Metin parçaları arası örtüşme
            
        Returns:
            İşlenmiş PDF verileri
        """
        if not self._validate_pdf_file(pdf_path):
            logger.error(f"Geçersiz PDF dosyası: {pdf_path}")
            return {}
        
        result = {
            "id": os.path.splitext(os.path.basename(pdf_path))[0],
            "source": os.path.basename(pdf_path),
            "category": category,
            "metadata": self.extract_metadata(pdf_path),
            "text": "",
            "tables": [],
            "table_text": "",
            "chunks": []
        }
        
        # Metni çıkar
        pdf_text = self.extract_text_with_pymupdf(pdf_path, max_pages)
        
        # Metin çıkartılamadıysa pdfplumber dene
        if not pdf_text.strip():
            pdf_text = self.extract_text_with_pdfplumber(pdf_path, max_pages)
            
        # Hala metin yoksa OCR uygula
        if not pdf_text.strip() and use_ocr and self.ocr_available:
            logger.info(f"Metinsel içerik bulunamadı, OCR uygulanıyor: {pdf_path}")
            pdf_text = self.perform_ocr(pdf_path)
        
        # Metni temizle
        pdf_text = self.clean_text(pdf_text)
        result["text"] = pdf_text
        
        # Tabloları çıkar
        if extract_tables:
            tables = self.extract_tables(pdf_path)
            result["tables"] = self.tables_to_json(tables)
            result["table_text"] = self.tables_to_text(tables)
        
        # Metni parçalara böl
        result["chunks"] = self.chunk_text(pdf_text, chunk_size, overlap)
        
        # Toplam kelime sayısı
        result["total_words"] = len(pdf_text.split())
        
        # İşlenmiş sonuçları kaydet
        output_path = self.output_dir / f"{result['id']}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"PDF başarıyla işlendi ve kaydedildi: {output_path}")
        return result
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Metni belirli uzunlukta parçalara böler.
        
        Args:
            text: Bölünecek metin
            chunk_size: Maksimum parça uzunluğu (kelime sayısı)
            overlap: Parçalar arası örtüşme (kelime sayısı)
            
        Returns:
            Metin parçalarının listesi
        """
        if not text:
            return []
            
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= chunk_size:
                current_chunk.append(word)
                current_length += len(word) + 1
            else:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "length": len(current_chunk),
                    "byte_size": len(chunk_text.encode('utf-8'))
                })
                
                # Örtüşme için son kelimeleri yeni chunk'a ekle
                overlap_words = min(overlap, len(current_chunk))
                current_chunk = current_chunk[-overlap_words:] + [word]
                current_length = sum(len(w) + 1 for w in current_chunk)
        
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "length": len(current_chunk),
                "byte_size": len(chunk_text.encode('utf-8'))
            })
        
        return chunks
    
    def process_pdf_directory(
        self, 
        directory: str, 
        category_mapping: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Bir dizindeki tüm PDF dosyalarını işler.
        
        Args:
            directory: PDF dosyalarının bulunduğu dizin
            category_mapping: Dosya adı deseni -> kategori eşleştirmeleri
            **kwargs: process_pdf'e geçirilecek ek parametreler
            
        Returns:
            İşlenmiş PDF verilerinin listesi
        """
        results = []
        pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
        
        for pdf_file in sorted(pdf_files):
            pdf_path = os.path.join(directory, pdf_file)
            
            # Kategoriyi belirle
            category = "genel"
            if category_mapping:
                for pattern, cat in category_mapping.items():
                    if pattern.lower() in pdf_file.lower():
                        category = cat
                        break
            
            try:
                # PDF'i işle
                result = self.process_pdf(pdf_path, category=category, **kwargs)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"PDF işlenirken hata: {pdf_file}, Hata: {str(e)}")
                continue
        
        logger.info(f"{len(results)} PDF dosyası başarıyla işlendi.")
        return results
    
    @staticmethod
    def process_pdf_to_documents(
        pdf_path: str,
        category: str = "genel",
        source: str = None,
        chunk_size: int = 1000,
        overlap: int = 200,
        max_pages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        PDF'i doğrudan belge listesine dönüştürür.
        
        Args:
            pdf_path: PDF dosya yolu
            category: Belge kategorisi
            source: Belge kaynağı (None ise dosya adı kullanılır)
            chunk_size: Metin parçalama uzunluğu
            overlap: Metin parçaları arası örtüşme
            max_pages: İşlenecek maksimum sayfa sayısı
            
        Returns:
            İşlenmiş belgelerin listesi
        """
        processor = EnhancedPdfProcessor()
        
        if not source:
            source = os.path.basename(pdf_path)
        
        pdf_data = processor.process_pdf(
            pdf_path,
            category=category,
            chunk_size=chunk_size,
            overlap=overlap,
            max_pages=max_pages
        )
        
        if not pdf_data:
            return []
        
        # Belgeleri oluştur
        documents = []
        
        # Ana metin belgeleri
        for i, chunk in enumerate(pdf_data["chunks"]):
            document_id = f"{pdf_data['id']}_chunk_{i}"
            documents.append({
                "id": document_id,
                "text": chunk["text"],
                "metadata": {
                    "source": source,
                    "category": category,
                    "chunk_index": i,
                    "total_chunks": len(pdf_data["chunks"]),
                    "page_range": pdf_data["metadata"].get("total_pages", "unknown"),
                    "document_title": pdf_data["metadata"].get("title", ""),
                    "type": "text_chunk"
                }
            })
        
        # Tablo belgeleri
        for i, table in enumerate(pdf_data.get("tables", [])):
            table_id = f"{pdf_data['id']}_table_{i}"
            table_text = f"Tablo {i+1}:\n"
            
            # Tablo başlıkları
            headers = table.get("headers", [])
            if headers:
                table_text += " | ".join(str(h) for h in headers) + "\n"
            
            # Tablo verileri
            for row in table.get("data", []):
                table_text += " | ".join(str(cell) if cell is not None else "" for cell in row.values()) + "\n"
            
            documents.append({
                "id": table_id,
                "text": table_text,
                "metadata": {
                    "source": source,
                    "category": category,
                    "table_index": i,
                    "total_tables": len(pdf_data.get("tables", [])),
                    "document_title": pdf_data["metadata"].get("title", ""),
                    "type": "table"
                }
            })
        
        return documents
    
    @staticmethod
    def process_pdf_directory_to_documents(
        directory: str,
        category_mapping: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Bir dizindeki tüm PDF'leri doğrudan belge listesine dönüştürür.
        
        Args:
            directory: PDF dosyalarının bulunduğu dizin
            category_mapping: Dosya adı deseni -> kategori eşleştirmeleri
            **kwargs: process_pdf_to_documents'e geçirilecek ek parametreler
            
        Returns:
            İşlenmiş belgelerin listesi
        """
        documents = []
        pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
        
        for pdf_file in sorted(pdf_files):
            pdf_path = os.path.join(directory, pdf_file)
            
            # Kategoriyi belirle
            category = "genel"
            if category_mapping:
                for pattern, cat in category_mapping.items():
                    if pattern.lower() in pdf_file.lower():
                        category = cat
                        break
            
            try:
                # PDF'i belgelere dönüştür
                pdf_documents = EnhancedPdfProcessor.process_pdf_to_documents(
                    pdf_path,
                    category=category,
                    source=pdf_file,
                    **kwargs
                )
                documents.extend(pdf_documents)
            except Exception as e:
                logger.error(f"PDF işlenirken hata: {pdf_file}, Hata: {str(e)}")
                continue
        
        logger.info(f"{len(pdf_files)} PDF dosyasından {len(documents)} belge oluşturuldu.")
        return documents
    
    def extract_text_with_pymupdf_streaming(self, pdf_path: str) -> Generator[str, None, None]:
        """
        PDF'ten PyMuPDF (fitz) kullanarak metin çıkarır (akış tabanlı).
        Her sayfa ayrı ayrı işlenir ve bellek verimliliği için yield edilir.
        
        Args:
            pdf_path: PDF dosya yolu
            
        Yields:
            Her sayfadan çıkarılan metin
        """
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(doc.page_count):
                # Bellek kullanımını kontrol et
                self._check_memory_usage()
                
                # Sayfayı yükle ve metni çıkar
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Sayfayı temizle
                page = None
                
                # Sayfadan çıkarılan metni yield et
                yield text
                
                # Her 10 sayfada bir belleği temizle
                if page_num % 10 == 0:
                    gc.collect()
            
            # Dökümanı kapat ve belleği temizle
            doc.close()
            doc = None
            gc.collect()
            
        except Exception as e:
            logger.error(f"PyMuPDF ile metin çıkarırken hata: {str(e)}")
            yield ""
    
    def extract_text_with_pdfplumber_streaming(self, pdf_path: str) -> Generator[str, None, None]:
        """
        PDF'ten pdfplumber kullanarak metin çıkarır (akış tabanlı).
        Her sayfa ayrı ayrı işlenir ve bellek verimliliği için yield edilir.
        
        Args:
            pdf_path: PDF dosya yolu
            
        Yields:
            Her sayfadan çıkarılan metin
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Bellek kullanımını kontrol et
                    self._check_memory_usage()
                    
                    # Metni çıkar
                    text = page.extract_text() or ""
                    
                    # Sayfadan çıkarılan metni yield et
                    yield text
                    
                    # Her 10 sayfada bir belleği temizle
                    if i % 10 == 0:
                        gc.collect()
        except Exception as e:
            logger.error(f"pdfplumber ile metin çıkarırken hata: {str(e)}")
            yield ""
    
    def process_pdf_streaming(
        self, 
        pdf_path: str, 
        category: str = "genel",
        output_path: Optional[str] = None,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> Dict[str, Any]:
        """
        Bir PDF dosyasını akış tabanlı yöntemle işler.
        Bu yöntem büyük PDF dosyaları için bellek verimlidir.
        
        Args:
            pdf_path: İşlenecek PDF dosya yolu
            category: Belge kategorisi
            output_path: Çıktı dosya yolu (None ise otomatik oluşturulur)
            chunk_size: Metin parçalama uzunluğu
            overlap: Metin parçaları arası örtüşme
            
        Returns:
            İşlenmiş PDF verileri özeti
        """
        if not self._validate_pdf_file(pdf_path):
            logger.error(f"Geçersiz PDF dosyası: {pdf_path}")
            return {}
        
        # Temel metadata oluştur
        metadata = self.extract_metadata(pdf_path)
        
        # Sonuç sözlüğünü başlat
        result = {
            "id": os.path.splitext(os.path.basename(pdf_path))[0],
            "source": os.path.basename(pdf_path),
            "category": category,
            "metadata": metadata,
            "chunk_count": 0,
            "total_words": 0,
            "processing_completed": False
        }
        
        # Çıktı yolunu belirle
        if not output_path:
            output_path = self.output_dir / f"{result['id']}.json"
        else:
            output_path = Path(output_path)
        
        try:
            # Akış işlemcisini başlat
            text_processor = TextStreamProcessor(
                output_path=output_path, 
                chunk_size=chunk_size, 
                overlap=overlap,
                cleanup_funcs=self.text_cleaners
            )
            
            # Akış işleme için sayfa sayfa oku
            page_count = 0
            word_count = 0
            
            # PyMuPDF ile metni çıkar
            for page_text in self.extract_text_with_pymupdf_streaming(pdf_path):
                if page_text.strip():
                    # Metni temizle ve işle
                    cleaned_text = self.clean_text(page_text)
                    chunks = text_processor.process_text_chunk(cleaned_text)
                    
                    page_count += 1
                    word_count += len(cleaned_text.split())
                
                # Düzenli bellek temizliği
                if page_count % 5 == 0:
                    self._check_memory_usage()
            
            # PyMuPDF ile metin bulunamazsa pdfplumber dene
            if page_count == 0:
                for page_text in self.extract_text_with_pdfplumber_streaming(pdf_path):
                    if page_text.strip():
                        # Metni temizle ve işle
                        cleaned_text = self.clean_text(page_text)
                        chunks = text_processor.process_text_chunk(cleaned_text)
                        
                        page_count += 1
                        word_count += len(cleaned_text.split())
                    
                    # Düzenli bellek temizliği
                    if page_count % 5 == 0:
                        self._check_memory_usage()
            
            # OCR işlemi akış tabanlı değil, bu yüzden sadece metin bulunamazsa uygula
            if page_count == 0 and self.ocr_available:
                logger.info(f"Metinsel içerik bulunamadı, OCR uygulanıyor: {pdf_path}")
                ocr_text = self.perform_ocr(pdf_path)
                if ocr_text.strip():
                    cleaned_text = self.clean_text(ocr_text)
                    chunks = text_processor.process_text_chunk(cleaned_text)
                    word_count = len(cleaned_text.split())
            
            # İşlemeyi tamamla
            chunk_count = text_processor.finalize()
            
            # Sonuç bilgilerini güncelle
            result["chunk_count"] = chunk_count
            result["total_words"] = word_count
            result["processing_completed"] = True
            
            # Özet bilgileri kaydet
            with open(str(output_path).replace('.json', '_summary.json'), 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"PDF akış tabanlı işleme tamamlandı: {pdf_path}")
            return result
            
        except Exception as e:
            logger.error(f"PDF akış tabanlı işleme hatası: {pdf_path}, Hata: {str(e)}")
            result["error"] = str(e)
            return result
    
    @force_gc_after
    def process_pdf_directory_streaming(
        self, 
        directory: str, 
        category_mapping: Optional[Dict[str, str]] = None,
        parallel: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Bir dizindeki tüm PDF dosyalarını akış tabanlı yöntemle işler.
        
        Args:
            directory: PDF dosyalarının bulunduğu dizin
            category_mapping: Dosya adı deseni -> kategori eşleştirmeleri
            parallel: Paralel işleme kullanılsın mı
            **kwargs: process_pdf_streaming'e geçirilecek ek parametreler
            
        Returns:
            İşlenmiş PDF verilerinin listesi
        """
        results = []
        pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
        
        if parallel and len(pdf_files) > 1:
            # Paralel işleme
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                for pdf_file in sorted(pdf_files):
                    pdf_path = os.path.join(directory, pdf_file)
                    
                    # Kategoriyi belirle
                    category = "genel"
                    if category_mapping:
                        for pattern, cat in category_mapping.items():
                            if pattern.lower() in pdf_file.lower():
                                category = cat
                                break
                    
                    # İşleme görevi ekle
                    future = executor.submit(
                        self.process_pdf_streaming,
                        pdf_path=pdf_path,
                        category=category,
                        **kwargs
                    )
                    futures.append(future)
                
                # Tüm görevlerin tamamlanmasını bekle
                for future in futures:
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Paralel PDF işleme hatası: {str(e)}")
        else:
            # Sıralı işleme
            for pdf_file in sorted(pdf_files):
                pdf_path = os.path.join(directory, pdf_file)
                
                # Kategoriyi belirle
                category = "genel"
                if category_mapping:
                    for pattern, cat in category_mapping.items():
                        if pattern.lower() in pdf_file.lower():
                            category = cat
                            break
                
                try:
                    # PDF'i işle
                    result = self.process_pdf_streaming(
                        pdf_path=pdf_path,
                        category=category,
                        **kwargs
                    )
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"PDF işlenirken hata: {pdf_file}, Hata: {str(e)}")
                    continue
                
                # Bellek temizliği
                self._check_memory_usage()
        
        logger.info(f"{len(results)} PDF dosyası akış tabanlı işleme ile başarıyla işlendi.")
        return results 