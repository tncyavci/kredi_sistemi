"""
PDF işleme performansını ölçen test modülü.
"""

import os
import time
import logging
import pytest
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import tempfile
from typing import List, Dict, Any

# Proje kök dizinini ekle
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing.enhanced_pdf_processor import EnhancedPdfProcessor
from utils.preprocessing.pdf_cache_manager import PDFCacheManager

# Loglama ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test verilerinin bulunduğu dizin
TEST_PDF_DIR = os.path.join(os.path.dirname(__file__), '..', 'test_pdfs')

class TestPDFProcessingPerformance:
    """PDF işleme performansını ölçen test sınıfı."""
    
    @pytest.fixture(scope="class")
    def setup_test_dirs(self):
        """Test dizinlerini oluştur"""
        # Geçici test dizinleri oluştur
        cache_dir = tempfile.mkdtemp(prefix="test_pdf_cache_")
        output_dir = tempfile.mkdtemp(prefix="test_pdf_output_")
        
        yield {"cache_dir": cache_dir, "output_dir": output_dir}
        
        # Temizlik
        shutil.rmtree(cache_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
    
    def get_test_pdf_files(self) -> List[str]:
        """Test edilecek PDF dosyalarını döndürür"""
        if not os.path.exists(TEST_PDF_DIR):
            pytest.skip(f"Test PDF dizini bulunamadı: {TEST_PDF_DIR}")
            
        pdf_files = [os.path.join(TEST_PDF_DIR, f) for f in os.listdir(TEST_PDF_DIR) 
                   if f.lower().endswith('.pdf')]
                   
        if not pdf_files:
            pytest.skip(f"Test PDF dizininde PDF dosyası bulunamadı: {TEST_PDF_DIR}")
            
        return pdf_files
    
    def test_pdf_processing_performance_comparison(self, setup_test_dirs):
        """
        Farklı ayarlarla PDF işleme performansını karşılaştırır:
        - GPU kullanarak ve kullanmadan
        - Önbellekleme ile ve olmadan
        - Farklı chunk boyutları ile
        """
        test_dirs = setup_test_dirs
        pdf_files = self.get_test_pdf_files()
        
        # Test senaryoları
        test_scenarios = [
            {"name": "CPU, No Cache", "use_gpu": False, "use_cache": False, "chunk_size": 1500},
            {"name": "CPU, With Cache", "use_gpu": False, "use_cache": True, "chunk_size": 1500},
            {"name": "GPU, No Cache", "use_gpu": True, "use_cache": False, "chunk_size": 1500},
            {"name": "GPU, With Cache", "use_gpu": True, "use_cache": True, "chunk_size": 1500},
            {"name": "GPU, Large Chunks", "use_gpu": True, "use_cache": False, "chunk_size": 3000},
        ]
        
        # Performans sonuçları
        results = []
        
        # Önbellek yöneticisi oluştur
        cache_manager = PDFCacheManager(
            cache_dir=test_dirs["cache_dir"],
            max_cache_size_mb=1000,
            use_compression=True
        )
        
        # Her senaryo için test yap
        for scenario in test_scenarios:
            # Önbelleği temizle
            if not scenario["use_cache"]:
                cache_manager.invalidate()
            
            processor = EnhancedPdfProcessor(
                output_dir=test_dirs["output_dir"],
                use_gpu=scenario["use_gpu"],
                gpu_batch_size=4
            )
            
            # En fazla 3 PDF dosyasını test et (çok uzun sürmesin)
            for pdf_file in pdf_files[:3]:
                pdf_name = os.path.basename(pdf_file)
                
                # Zamanı ölç
                start_time = time.time()
                
                # PDF'i işle
                documents = processor.process_pdf_to_documents(
                    pdf_path=pdf_file,
                    category="test",
                    chunk_size=scenario["chunk_size"],
                    overlap=100,
                    use_ocr=True,
                    extract_tables=True,
                    use_gpu=scenario["use_gpu"],
                    use_cache=scenario["use_cache"],
                    force_refresh=False
                )
                
                elapsed_time = time.time() - start_time
                
                # Sonuçları kaydet
                results.append({
                    "scenario": scenario["name"],
                    "pdf_name": pdf_name,
                    "elapsed_time": elapsed_time,
                    "document_count": len(documents),
                    "use_gpu": scenario["use_gpu"],
                    "use_cache": scenario["use_cache"],
                    "chunk_size": scenario["chunk_size"]
                })
                
                logger.info(f"Senaryo: {scenario['name']}, PDF: {pdf_name}, Süre: {elapsed_time:.2f} saniye, Belgeler: {len(documents)}")
        
        # Sonuçları göster
        if results:
            df = pd.DataFrame(results)
            print("\nPDF İşleme Performans Sonuçları:")
            print(df[["scenario", "pdf_name", "elapsed_time", "document_count"]])
            
            # Ortalama süreleri hesapla
            avg_times = df.groupby("scenario")["elapsed_time"].mean()
            print("\nOrtalama İşleme Süreleri (saniye):")
            print(avg_times)
            
            # Önbellek istatistikleri
            cache_stats = cache_manager.get_cache_stats()
            print("\nÖnbellek İstatistikleri:")
            for key, value in cache_stats.items():
                print(f"{key}: {value}")
                
            # Sonuçları görselleştir
            plt.figure(figsize=(12, 6))
            avg_times.plot(kind='bar')
            plt.title('PDF İşleme Performans Karşılaştırması')
            plt.ylabel('İşleme Süresi (saniye)')
            plt.xlabel('Senaryo')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Grafiği kaydet
            plot_path = os.path.join(test_dirs["output_dir"], "performance_comparison.png")
            plt.savefig(plot_path)
            print(f"\nPerformans grafiği kaydedildi: {plot_path}")
        
        assert len(results) > 0, "Hiç performans testi sonucu alınamadı"
    
    def test_cache_efficiency(self, setup_test_dirs):
        """Önbellekleme verimliliğini test eder"""
        test_dirs = setup_test_dirs
        pdf_files = self.get_test_pdf_files()
        
        if not pdf_files:
            pytest.skip("Test için PDF dosyası bulunamadı")
            
        # Test için ilk PDF'i kullan
        pdf_file = pdf_files[0]
        
        # Önbellek yöneticisi oluştur
        cache_manager = PDFCacheManager(
            cache_dir=test_dirs["cache_dir"],
            max_cache_size_mb=1000,
            use_compression=True
        )
        
        # Önbelleği temizle
        cache_manager.invalidate()
        
        processor = EnhancedPdfProcessor(
            output_dir=test_dirs["output_dir"],
            use_gpu=False
        )
        
        # İlk çalıştırma süresi (önbelleksiz)
        start_time = time.time()
        processor.process_pdf_to_documents(
            pdf_path=pdf_file,
            category="test",
            use_cache=True,
            force_refresh=True
        )
        first_run_time = time.time() - start_time
        
        # İkinci çalıştırma süresi (önbellekli)
        start_time = time.time()
        processor.process_pdf_to_documents(
            pdf_path=pdf_file,
            category="test",
            use_cache=True,
            force_refresh=False
        )
        second_run_time = time.time() - start_time
        
        # Önbellek hızlanma oranını hesapla
        speedup = first_run_time / second_run_time if second_run_time > 0 else float('inf')
        
        print(f"\nÖnbellek Verimlilik Testi:")
        print(f"PDF: {os.path.basename(pdf_file)}")
        print(f"İlk çalıştırma süresi: {first_run_time:.2f} saniye")
        print(f"İkinci çalıştırma süresi (önbellekli): {second_run_time:.2f} saniye")
        print(f"Hızlanma oranı: {speedup:.2f}x")
        
        # Önbellekleme en az 5x hızlanma sağlamalı
        assert speedup > 5, f"Önbellekleme yeterince verimli değil: sadece {speedup:.2f}x hızlanma"
    

if __name__ == "__main__":
    # Doğrudan çalıştırma için
    test_instance = TestPDFProcessingPerformance()
    dirs = test_instance.setup_test_dirs().__next__()
    test_instance.test_pdf_processing_performance_comparison(dirs)
    test_instance.test_cache_efficiency(dirs) 