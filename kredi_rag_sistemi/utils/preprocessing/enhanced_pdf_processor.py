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
import time
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

# GPU kontrolü için
import torch

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
        max_workers: int = 4,  # Paralel işleme için maksimum iş parçacığı sayısı
        use_gpu: bool = None,  # GPU kullanılsın mı (None: otomatik tespit)
        gpu_batch_size: int = 4  # GPU için batch boyutu
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
            use_gpu: GPU kullanılsın mı (None: otomatik tespit)
            gpu_batch_size: GPU için batch boyutu
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.ocr_lang = ocr_lang
        self.ocr_threshold = ocr_threshold
        self.table_extraction_method = table_extraction_method
        self.memory_threshold_mb = memory_threshold_mb
        self.use_streaming = use_streaming
        self.max_workers = max_workers
        self.gpu_batch_size = gpu_batch_size
        
        # GPU kullanım durumunu kontrol et
        if use_gpu is None:
            # CUDA veya MPS varsa GPU'yu otomatik olarak etkinleştir
            self.use_gpu = torch.cuda.is_available() or hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            if self.use_gpu:
                logger.info(f"GPU otomatik olarak tespit edildi. Kullanılabilir: {self.use_gpu}")
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    logger.info(f"CUDA cihazı kullanılıyor: {torch.cuda.get_device_name(0)}")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                    logger.info("Apple M1/M2 MPS cihazı kullanılıyor")
        else:
            self.use_gpu = use_gpu
            if self.use_gpu:
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                else:
                    logger.warning("GPU istendi ancak kullanılabilir GPU bulunamadı. CPU'ya geri dönülüyor.")
                    self.use_gpu = False
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device("cpu")
        
        # GPU kullanımı hakkında bilgi ver
        logger.info(f"GPU kullanımı: {'Aktif' if self.use_gpu else 'Devre dışı'}, Cihaz: {self.device}")
        
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
            
        # GPU destekli OCR için EasyOCR'ı kontrol et
        self.easyocr_available = False
        if self.use_gpu:
            try:
                import easyocr
                self.easyocr_reader = easyocr.Reader(['tr', 'en'])
                self.easyocr_available = True
                logger.info("EasyOCR GPU desteği ile kullanılabilir.")
            except ImportError:
                logger.warning("EasyOCR bulunamadı, GPU destekli OCR devre dışı.")
                self.easyocr_available = False
    
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
            max_file_size_mb = 500  # 500MB üst sınır (daha önce 100MB idi)
            
            if file_size_mb > max_file_size_mb:
                logger.warning(f"PDF dosyası çok büyük: {file_path} ({file_size_mb:.2f} MB), maksimum: {max_file_size_mb} MB")
                return False
                
            logger.info(f"PDF dosya boyutu: {file_size_mb:.2f} MB")
            
            # PDF'in açılabilirliğini kontrol et
            with open(file_path, 'rb') as f:
                # PyPDF2 kullanarak PDF'in geçerliliğini kontrol et
                try:
                    pdf = PyPDF2.PdfReader(f)
                    # Sayfa sayısını kontrol et (çok fazla sayfa varsa uyarı ver)
                    page_count = len(pdf.pages)
                    if page_count > 5000:
                        logger.warning(f"PDF dosyası çok fazla sayfa içeriyor: {file_path} ({page_count} sayfa)")
                        return False
                        
                    logger.info(f"PDF sayfa sayısı: {page_count}")
                    
                    # PDF'in şifreli olup olmadığını kontrol et
                    if pdf.is_encrypted:
                        logger.warning(f"PDF dosyası şifreli: {file_path}")
                        return False
                    
                    return True
                except Exception as e:
                    # PyPDF2 başarısız olursa PyMuPDF ile dene
                    logger.warning(f"PyPDF2 ile doğrulama başarısız, PyMuPDF deneniyor: {str(e)}")
                    
            # PyMuPDF ile dene
            try:
                doc = fitz.open(file_path)
                page_count = doc.page_count
                
                if page_count > 5000:
                    logger.warning(f"PDF dosyası çok fazla sayfa içeriyor: {file_path} ({page_count} sayfa)")
                    doc.close()
                    return False
                
                # Şifreli mi kontrol et
                if doc.isEncrypted:
                    logger.warning(f"PDF dosyası şifreli: {file_path}")
                    doc.close()
                    return False
                
                doc.close()
                return True
            except Exception as e:
                logger.error(f"PDF doğrulama hatası (PyMuPDF): {str(e)}")
                return False
                
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
    
    def perform_gpu_accelerated_ocr(self, images: List[np.ndarray]) -> List[str]:
        """
        Görüntü listesi üzerinde GPU hızlandırmalı OCR gerçekleştirir
        
        Args:
            images: İşlenecek görüntü dizisi (numpy array formatında)
            
        Returns:
            OCR sonucu metinler listesi
        """
        if not self.easyocr_available:
            logger.warning("EasyOCR yüklü değil, Tesseract'a geri dönülüyor")
            return [pytesseract.image_to_string(img, lang=self.ocr_lang) for img in images]
            
        try:
            import easyocr
            
            batch_results = []
            # Batch'ler halinde işle (bellek için)
            for i in range(0, len(images), self.gpu_batch_size):
                batch = images[i:i+self.gpu_batch_size]
                
                # GPU için bellek optimizasyonu
                if i % (self.gpu_batch_size * 3) == 0:
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                
                # Batch OCR işlemi
                if self.use_gpu:
                    with torch.no_grad():  # Gradyan hesaplamalarını devre dışı bırak
                        batch_texts = []
                        for img in batch:
                            # EasyOCR ile OCR işlemi
                            try:
                                result = self.easyocr_reader.readtext(img)
                                text = " ".join([entry[1] for entry in result])
                                batch_texts.append(text)
                            except Exception as e:
                                logger.error(f"GPU OCR hatası: {str(e)}")
                                # Hata durumunda Tesseract'a geri dön
                                text = pytesseract.image_to_string(img, lang=self.ocr_lang)
                                batch_texts.append(text)
                else:
                    # GPU kullanılamazsa Tesseract'a geri dön
                    batch_texts = [pytesseract.image_to_string(img, lang=self.ocr_lang) for img in batch]
                    
                batch_results.extend(batch_texts)
                
            return batch_results
            
        except Exception as e:
            logger.error(f"GPU OCR genel hatası: {str(e)}")
            # Sorun durumunda geleneksel OCR'a geri dön
            return [pytesseract.image_to_string(img, lang=self.ocr_lang) for img in images]

    def preprocess_images_for_ocr(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        OCR öncesinde görüntüleri iyileştirmek için ön işleme yapar
        
        Args:
            images: İşlenecek görüntü dizisi
            
        Returns:
            İyileştirilmiş görüntü dizisi
        """
        processed_images = []
        
        for img in images:
            # Griye dönüştür
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Gürültüyü gider (Gaussian blur)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Kontrastı artır (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(blurred)
            
            # Adaptif eşikleme
            binary = cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2
            )
            
            # Morfolojik işlemler
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # OCR için yeniden tersine çevir (siyah yazı beyaz zemin)
            final = cv2.bitwise_not(closing)
            
            # Dokumanın eğikliğini düzelt
            coords = np.column_stack(np.where(final > 0))
            if len(coords) > 0:
                try:
                    angle = cv2.minAreaRect(coords)[-1]
                    if angle < -45:
                        angle = -(90 + angle)
                    else:
                        angle = -angle
                        
                    # Eğer belirgin bir eğiklik varsa düzelt
                    if abs(angle) > 0.5:
                        (h, w) = final.shape[:2]
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        final = cv2.warpAffine(final, M, (w, h), 
                                              flags=cv2.INTER_CUBIC, 
                                              borderMode=cv2.BORDER_REPLICATE)
                except Exception as e:
                    logger.warning(f"Görüntü eğikliği düzeltme hatası: {str(e)}")
            
            processed_images.append(final)
            
        return processed_images

    def perform_ocr(self, pdf_path: str, page_numbers: Optional[List[int]] = None) -> str:
        """
        PDF üzerinde OCR gerçekleştirir.
        
        Args:
            pdf_path: OCR yapılacak PDF dosyasının yolu
            page_numbers: OCR yapılacak sayfaların listesi (None ise tüm sayfalar)
            
        Returns:
            OCR işlemi sonucu metin
        """
        if not self.ocr_available and not self.easyocr_available:
            logger.error("OCR işlemi gerçekleştirilemiyor: OCR kütüphanesi bulunamadı.")
            return ""
            
        try:
            # PDF'i görüntülere dönüştür
            logger.info(f"PDF görüntülere dönüştürülüyor: {pdf_path}")
            images = convert_from_path(pdf_path, 300)  # 300 DPI
            
            # Belirli sayfalar seçildiyse sadece o sayfaları işle
            if page_numbers:
                selected_images = [images[i-1] for i in page_numbers if 0 < i <= len(images)]
                if not selected_images:
                    logger.warning(f"Belirtilen sayfa numaraları PDF'te bulunamadı: {page_numbers}")
                    selected_images = images
            else:
                selected_images = images
                
            # Bellek verimliliği için atık görüntüleri temizle
            if selected_images != images:
                del images
                gc.collect()
            
            logger.info(f"Toplam {len(selected_images)} sayfa OCR işlemi için hazırlanıyor")
            
            # Görüntüleri numpy array'lere dönüştür
            np_images = []
            for img in selected_images:
                # PIL görüntüsünü numpy array'e dönüştür
                np_img = np.array(img)
                np_images.append(np_img)
                
            # Bellek verimliliği için orijinal görüntüleri temizle
            del selected_images
            gc.collect()
            
            # Görüntüleri iyileştir
            logger.info("Görüntüler OCR öncesi iyileştiriliyor")
            processed_images = self.preprocess_images_for_ocr(np_images)
            
            # Bellek verimliliği için orijinal görüntüleri temizle
            del np_images
            gc.collect()
            
            # OCR işlemini gerçekleştir
            ocr_results = []
            
            # GPU destekli OCR kullanılabiliyorsa
            if self.use_gpu and self.easyocr_available:
                logger.info("GPU destekli OCR başlatılıyor")
                
                # Görüntüleri batches halinde işle
                ocr_texts = self.perform_gpu_accelerated_ocr(processed_images)
                ocr_results.extend(ocr_texts)
            else:
                # Standart Tesseract OCR kullan
                logger.info("Tesseract OCR başlatılıyor")
                for img in processed_images:
                    # Görüntüyü OCR ile işle
                    text = pytesseract.image_to_string(img, lang=self.ocr_lang)
                    ocr_results.append(text)
            
            # Bellek temizleme
            del processed_images
            gc.collect()
            
            # Sonuçları birleştir
            final_text = "\n\n".join(ocr_results)
            
            # Metni temizle
            final_text = self.clean_text(final_text)
            
            logger.info(f"OCR tamamlandı, {len(ocr_results)} sayfa metin çıkarıldı")
            
            return final_text
            
        except Exception as e:
            logger.error(f"OCR işlemi sırasında hata: {str(e)}")
            return ""
    
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
            # Önce lattice (çizgili tablolar) dene
            logger.info(f"Camelot lattice ile tablo çıkarma deneniyor: {pdf_path}")
            extracted_tables = camelot.read_pdf(
                pdf_path,
                pages=pages,
                flavor='lattice',  # Çizgiler ve sınırlar olan tablolar için
                suppress_stdout=True,
                line_scale=40,  # Çizgileri daha iyi tespit etmek için
                strip_text='\n' # Satır içi yeni satırları temizle
            )
            
            logger.info(f"Camelot lattice ile {len(extracted_tables)} tablo bulundu.")
            
            # Kaliteli ve kullanılabilir tabloları filtrele
            for i, table in enumerate(extracted_tables):
                if table.accuracy > 60:  # Eşiği düşürdük
                    tables.append(table.df)
                    
            # Eğer hiç tablo bulunamazsa başka flavor dene
            if not tables:
                logger.info(f"Camelot stream ile tablo çıkarma deneniyor: {pdf_path}")
                # Stream flavor ile dene (çizgisiz tablolar)
                extracted_tables = camelot.read_pdf(
                    pdf_path,
                    pages=pages,
                    flavor='stream',  # Çizgisiz tablolar için
                    suppress_stdout=True,
                    strip_text='\n',
                    edge_tol=500 # Daha esnek kenar tespiti
                )
                
                logger.info(f"Camelot stream ile {len(extracted_tables)} tablo bulundu.")
                
                for i, table in enumerate(extracted_tables):
                    if table.accuracy > 50:  # Stream için daha düşük eşik
                        tables.append(table.df)
        except Exception as e:
            logger.error(f"Camelot ile tablo çıkarırken hata: {str(e)}")
            
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
            logger.info(f"Tabula ile tablo çıkarma deneniyor: {pdf_path}")
            
            # Önce automatic detection dene
            tables = tabula.read_pdf(
                pdf_path,
                pages=pages,
                multiple_tables=True
            )
            
            logger.info(f"Tabula ile {len(tables)} tablo bulundu.")
            
            # Eğer tablo bulunamazsa, daha agresif detection dene
            if not tables:
                logger.info("Tabula guess=True ile tekrar deneniyor...")
                tables = tabula.read_pdf(
                    pdf_path,
                    pages=pages,
                    multiple_tables=True,
                    guess=True,
                    pandas_options={'header': None}
                )
                logger.info(f"Tabula guess=True ile {len(tables)} tablo bulundu.")
                
            # Boş tablolar varsa kaldır
            return [table for table in tables if not table.empty]
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
        if not tables:
            return ""
            
        text = ""
        for i, table in enumerate(tables):
            # Tablo başlık satırı  
            text += f"\n**TABLO {i+1}**\n"
            
            # NaN değerleri boş stringlerle değiştir
            table = table.fillna("")
            
            # Başlık satırını daha belirgin yap
            if not table.empty:
                headers = list(table.columns)
                # Başlıkları büyük harfle yaz
                header_row = " | ".join([str(h).strip().upper() for h in headers])
                text += header_row + "\n"
                # Alt çizgi ekle
                text += "-" * len(header_row) + "\n"
                
                # Veri satırları
                for _, row in table.iterrows():
                    # Her hücreyi temizle ve formatlı göster
                    row_values = []
                    for val in row:
                        if isinstance(val, (float, int)):
                            # Sayısal değerleri düzgün formatla
                            row_values.append(f"{val:,}".replace(",", "."))
                        else:
                            # Metin değerlerini temizle
                            str_val = str(val).strip().replace("\n", " ")
                            row_values.append(str_val)
                    text += " | ".join(row_values) + "\n"
                
                text += "\n"  # Tablolar arası boşluk
        
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
            # Boş tabloları atla
            if table.empty:
                continue
                
            # NaN değerleri boş stringlerle değiştir
            table = table.fillna("")
            
            # Tablo verilerini düzenle
            headers = [str(h).strip() for h in table.columns]
            records = []
            
            for _, row in table.iterrows():
                record = {}
                for j, header in enumerate(headers):
                    # İndeks kullan çünkü bazı başlıklar boş olabilir
                    value = row.iloc[j]
                    
                    # Değeri uygun formata dönüştür
                    if isinstance(value, (float, int)) and not pd.isna(value):
                        # Sayısal değerleri düzgün formatla
                        value = float(value) if isinstance(value, float) else int(value)
                    elif isinstance(value, str):
                        # Metin değerlerini temizle
                        value = value.strip().replace("\n", " ")
                    else:
                        value = str(value)
                        
                    # Boş başlık durumunu ele al
                    if not header:
                        header = f"Column_{j+1}"
                        
                    record[header] = value
                    
                records.append(record)
            
            # Anlamlı tablo bilgisi oluştur
            table_info = {
                "table_id": i+1,
                "table_name": f"Tablo_{i+1}",
                "headers": headers,
                "row_count": len(records),
                "column_count": len(headers),
                "data": records
            }
            
            result.append(table_info)
            
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
        overlap: int = 200,
        use_cache: bool = True,
        force_refresh: bool = False,
        cache_manager = None
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
            use_cache: Önbellek kullanılsın mı
            force_refresh: Önbelleği zorla yenile
            cache_manager: Kullanılacak önbellek yöneticisi
            
        Returns:
            İşlenmiş PDF verileri
        """
        if not self._validate_pdf_file(pdf_path):
            logger.error(f"Geçersiz PDF dosyası: {pdf_path}")
            return {}
            
        # Önbellekleme için işleme parametrelerini hazırla
        process_params = {
            "category": category,
            "extract_tables": extract_tables,
            "use_ocr": use_ocr,
            "max_pages": max_pages,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "use_gpu": self.use_gpu,
            "gpu_batch_size": self.gpu_batch_size
        }
        
        # Önbellekten yükleme
        if use_cache and not force_refresh:
            try:
                from utils.preprocessing.pdf_cache_manager import PDFCacheManager
                
                # Önbellek yöneticisi oluştur veya var olanı kullan
                cache_mgr = cache_manager or PDFCacheManager()
                
                # Önbellekten PDF sonucunu al
                cached_result = cache_mgr.get(pdf_path, process_params)
                if cached_result:
                    logger.info(f"PDF dosyası önbellekten yüklendi: {pdf_path}")
                    return cached_result
            except ImportError:
                logger.warning("Önbellek yöneticisi bulunamadı, önbellekleme atlanıyor")
            except Exception as e:
                logger.warning(f"Önbellekten yükleme hatası: {str(e)}")
        
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
        
        # Başlangıç zamanını kaydet
        start_time = time.time()
        
        # Metni çıkar
        pdf_text = self.extract_text_with_pymupdf(pdf_path, max_pages)
        
        # Metin çıkartılamadıysa pdfplumber dene
        if not pdf_text.strip():
            pdf_text = self.extract_text_with_pdfplumber(pdf_path, max_pages)
            
        # Hala metin yoksa OCR uygula
        if not pdf_text.strip() and use_ocr:
            logger.info(f"Metinsel içerik bulunamadı, OCR uygulanıyor: {pdf_path}")
            if self.use_gpu and self.easyocr_available:
                logger.info(f"GPU destekli OCR kullanılıyor: {pdf_path}")
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
        
        # İşleme süresini hesapla
        processing_time = time.time() - start_time
        result["processing_info"] = {
            "processing_time_seconds": processing_time,
            "use_gpu": self.use_gpu,
            "ocr_used": use_ocr and not pdf_text.strip(),
            "ocr_engine": "easyocr_gpu" if self.use_gpu and self.easyocr_available else "tesseract",
            "tables_extracted": extract_tables,
            "total_tables": len(result["tables"]),
            "total_chunks": len(result["chunks"]),
            "chunk_size": chunk_size,
            "overlap": overlap
        }
        
        # İşlenmiş sonuçları kaydet
        output_path = self.output_dir / f"{result['id']}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"PDF başarıyla işlendi ve kaydedildi: {output_path} ({processing_time:.2f} saniye)")
        
        # Önbelleğe kaydet
        if use_cache:
            try:
                from utils.preprocessing.pdf_cache_manager import PDFCacheManager
                
                # Önbellek yöneticisi oluştur veya var olanı kullan
                cache_mgr = cache_manager or PDFCacheManager()
                
                # Sonucu önbelleğe kaydet
                cache_mgr.set(pdf_path, process_params, result)
            except ImportError:
                logger.warning("Önbellek yöneticisi bulunamadı, önbellekleme atlanıyor")
            except Exception as e:
                logger.warning(f"Önbelleğe kaydetme hatası: {str(e)}")
        
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
        max_pages: Optional[int] = None,
        use_ocr: bool = True,
        extract_tables: bool = True,
        use_gpu: bool = None,
        gpu_batch_size: int = 4,
        use_cache: bool = True,
        force_refresh: bool = False,
        table_extraction_method: str = "auto",
        prioritize_tables: bool = False,
        keep_table_context: bool = False
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
            use_ocr: OCR kullanılsın mı
            extract_tables: Tablolar çıkarılsın mı
            use_gpu: GPU kullanılsın mı (None ise otomatik tespit)
            gpu_batch_size: GPU için batch boyutu
            use_cache: Önbellek kullanılsın mı
            force_refresh: Önbelleği zorla yenile
            table_extraction_method: Tablo çıkarma yöntemi ('auto', 'camelot', 'tabula')
            prioritize_tables: Tablolar metin parçalarından önce gelsin mi
            keep_table_context: Tabloları bağlamlarıyla birlikte tut
            
        Returns:
            İşlenmiş belgelerin listesi
        """
        try:
            # Önbellekleme için cache manager oluştur
            from utils.preprocessing.pdf_cache_manager import PDFCacheManager
            cache_manager = PDFCacheManager()
        except ImportError:
            logger.info("Önbellek yöneticisi bulunamadı, önbellekleme devre dışı.")
            cache_manager = None
            use_cache = False
        
        # GPU destekli işlemci oluştur
        processor = EnhancedPdfProcessor(
            use_gpu=use_gpu,
            gpu_batch_size=gpu_batch_size,
            table_extraction_method=table_extraction_method
        )
        
        if not source:
            source = os.path.basename(pdf_path)
        
        # Önbellekleme özelliğini belirterek PDF'i işle
        pdf_data = processor.process_pdf(
            pdf_path,
            category=category,
            chunk_size=chunk_size,
            overlap=overlap,
            max_pages=max_pages,
            use_ocr=use_ocr,
            extract_tables=extract_tables,
            use_cache=use_cache,
            force_refresh=force_refresh,
            cache_manager=cache_manager
        )
        
        if not pdf_data:
            return []
        
        # Belgeleri oluştur
        documents = []
        
        # Eğer tablolar önceliklendirilecekse, önce tablo belgelerini ekle
        if prioritize_tables and extract_tables:
            # Tablo belgeleri
            for i, table in enumerate(pdf_data.get("tables", [])):
                table_id = f"{pdf_data['id']}_table_{i}"
                table_text = f"TABLO {i+1}:\n"
                
                # Tablo başlıkları
                headers = table.get("headers", [])
                if headers:
                    table_text += " | ".join(str(h) for h in headers) + "\n"
                    # Alt çizgi ekle
                    table_text += "-" * len(table_text.split("\n")[-2]) + "\n"
                
                # Tablo verileri
                for row in table.get("data", []):
                    table_text += " | ".join(str(cell) if cell is not None else "" for cell in row.values()) + "\n"
                
                # Eğer tablonun bağlamını korumak isteniyorsa
                if keep_table_context:
                    # Bağlam ekle (tablonun bulunduğu yeri göster)
                    context_text = ""
                    chunks = pdf_data.get("chunks", [])
                    for chunk in chunks:
                        chunk_text = chunk.get("text", "")
                        # Eğer tabloya referans varsa
                        if f"Tablo {i+1}" in chunk_text or f"tablo {i+1}" in chunk_text.lower():
                            context_text += chunk_text + "\n\n"
                    
                    if context_text:
                        table_text += "\nBAĞLAM:\n" + context_text
                
                documents.append({
                    "id": table_id,
                    "text": table_text,
                    "metadata": {
                        "source": source,
                        "category": category,
                        "table_index": i,
                        "total_tables": len(pdf_data.get("tables", [])),
                        "document_title": pdf_data["metadata"].get("title", ""),
                        "type": "table",
                        "file_path": pdf_path,
                        "processing_time": pdf_data.get("processing_info", {}).get("processing_time_seconds", 0)
                    }
                })
        
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
                    "type": "text_chunk",
                    "chunk_size": chunk_size,
                    "file_path": pdf_path,
                    "processing_type": "gpu_ocr" if processor.use_gpu else "cpu_ocr" if use_ocr else "text_extraction",
                    "processing_time": pdf_data.get("processing_info", {}).get("processing_time_seconds", 0),
                    "ocr_engine": pdf_data.get("processing_info", {}).get("ocr_engine", "")
                }
            })
        
        # Tablolar önceliklendirilmemişse, şimdi tablo belgelerini ekle
        if not prioritize_tables and extract_tables:
            # Yukarıdaki tablo ekleme kodunu buraya da tekrarla
            for i, table in enumerate(pdf_data.get("tables", [])):
                table_id = f"{pdf_data['id']}_table_{i}"
                table_text = f"TABLO {i+1}:\n"
                
                # Tablo başlıkları
                headers = table.get("headers", [])
                if headers:
                    table_text += " | ".join(str(h) for h in headers) + "\n"
                    # Alt çizgi ekle
                    table_text += "-" * len(table_text.split("\n")[-2]) + "\n"
                
                documents.append({
                    "id": table_id,
                    "text": table_text,
                    "metadata": {
                        "source": source,
                        "category": category,
                        "table_index": i,
                        "total_tables": len(pdf_data.get("tables", [])),
                        "document_title": pdf_data["metadata"].get("title", ""),
                        "type": "table",
                        "file_path": pdf_path,
                        "processing_time": pdf_data.get("processing_info", {}).get("processing_time_seconds", 0)
                    }
                })
        
        # İşleme durumunu logla
        processing_info = pdf_data.get("processing_info", {})
        if processing_info:
            logger.info(
                f"PDF işleme tamamlandı: {pdf_path}, "
                f"Süre: {processing_info.get('processing_time_seconds', 0):.2f} saniye, "
                f"OCR: {processing_info.get('ocr_used', False)}, "
                f"GPU: {processing_info.get('use_gpu', False)}, "
                f"Belge sayısı: {len(documents)}"
            )
        
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