import os
import pytest
import tempfile
import time
import json
import gc
import shutil
import psutil
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from utils.preprocessing.enhanced_pdf_processor import EnhancedPdfProcessor, get_memory_usage

# PyTest marker for performance tests
pytestmark = pytest.mark.performance

@pytest.fixture
def test_pdf_dir():
    """Test PDF dizini"""
    test_dir = Path(__file__).parent.parent / "test_pdfs"
    if not test_dir.exists():
        pytest.skip("Test PDF'leri bulunamadı. Önce 'create_test_pdfs.py' çalıştırılmalı.")
    return str(test_dir)

@pytest.fixture
def large_test_pdf(test_pdf_dir):
    """
    Performans testleri için büyük PDF oluştur.
    Bu test, varolan PDF'leri birleştirerek daha büyük bir test dosyası oluşturur.
    """
    pdf_files = [f for f in os.listdir(test_pdf_dir) if f.lower().endswith('.pdf')]
    if len(pdf_files) < 1:
        pytest.skip("Test PDF'leri bulunamadı.")
        
    # Geçici dosya oluştur
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        large_pdf_path = tmp.name
    
    # PyPDF2 ile PDF'leri birleştir
    try:
        import PyPDF2
        writer = PyPDF2.PdfWriter()
        
        # Her PDF'i 5 kez ekleyerek daha büyük dosya oluştur
        for pdf_file in pdf_files:
            pdf_path = os.path.join(test_pdf_dir, pdf_file)
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for _ in range(5):  # Her PDF'i 5 kez ekle
                    for page in reader.pages:
                        writer.add_page(page)
        
        # Birleştirilmiş PDF'i yaz
        with open(large_pdf_path, 'wb') as f:
            writer.write(f)
        
        yield large_pdf_path
        
        # Test sonunda geçici dosyayı temizle
        if os.path.exists(large_pdf_path):
            os.unlink(large_pdf_path)
            
    except ImportError:
        pytest.skip("PyPDF2 bulunamadı.")

@pytest.fixture
def output_dir():
    """Geçici çıktı dizini"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

class TestPdfProcessorPerformance:
    """EnhancedPdfProcessor'ın performans testleri"""
    
    def test_processing_time(self, test_pdf_dir, output_dir):
        """PDF işleme süresini ölçen performans testi"""
        processor = EnhancedPdfProcessor(output_dir=output_dir)
        
        pdf_files = [f for f in os.listdir(test_pdf_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            pytest.skip("Test PDF'leri bulunamadı.")
        
        pdf_path = os.path.join(test_pdf_dir, pdf_files[0])
        
        # İşleme süresini ölç
        start_time = time.time()
        result = processor.process_pdf(pdf_path)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Süreyi logla
        print(f"\nPDF işleme süresi: {processing_time:.2f} saniye")
        
        # Çok uzun sürmemeli (gerçekçi bir üst sınır belirleyin)
        assert processing_time < 30, f"PDF işleme çok uzun sürdü: {processing_time:.2f} saniye"
    
    def test_memory_usage(self, test_pdf_dir, output_dir):
        """Bellek kullanımını ölçen performans testi"""
        # GC çalıştırarak başlangıç bellek durumunu temizle
        gc.collect()
        
        # Başlangıç bellek kullanımı
        start_memory = get_memory_usage()
        
        # PDF'i işle
        processor = EnhancedPdfProcessor(output_dir=output_dir)
        
        pdf_files = [f for f in os.listdir(test_pdf_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            pytest.skip("Test PDF'leri bulunamadı.")
        
        pdf_path = os.path.join(test_pdf_dir, pdf_files[0])
        
        # İşle
        result = processor.process_pdf(pdf_path)
        
        # İşlem sonrası bellek kullanımı
        gc.collect()  # Kullanılmayan nesneleri temizle
        end_memory = get_memory_usage()
        
        # Bellek kullanımı artışı
        memory_increase = end_memory - start_memory
        
        # Bellek kullanımını logla
        print(f"\nBellek kullanımı artışı: {memory_increase:.2f} MB")
        
        # Çok fazla bellek kullanmamalı (gerçekçi bir üst sınır belirleyin)
        assert memory_increase < 200, f"Bellek kullanımı çok yüksek: {memory_increase:.2f} MB"
    
    def test_streaming_vs_standard_memory(self, large_test_pdf, output_dir):
        """Akış tabanlı ve standart işlemenin bellek kullanımını karşılaştıran test"""
        
        # Bellek kullanımını izlemek için fonksiyon
        def get_peak_memory_usage(func, *args, **kwargs):
            gc.collect()
            process = psutil.Process(os.getpid())
            
            # Başlangıç bellek kullanımı
            start_memory = process.memory_info().rss / (1024 * 1024)
            
            # Bellek kullanımını izle
            peak_memory = start_memory
            stop_monitoring = False
            
            def memory_monitor():
                nonlocal peak_memory
                while not stop_monitoring:
                    current_memory = process.memory_info().rss / (1024 * 1024)
                    peak_memory = max(peak_memory, current_memory)
                    time.sleep(0.1)
            
            # İzleme thread'ini başlat
            import threading
            monitor_thread = threading.Thread(target=memory_monitor)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            try:
                # Fonksiyonu çalıştır
                result = func(*args, **kwargs)
                return result, peak_memory - start_memory
            finally:
                # İzleme thread'ini durdur
                stop_monitoring = True
                monitor_thread.join(timeout=1.0)
                gc.collect()
        
        # Standart işleme
        processor_standard = EnhancedPdfProcessor(
            output_dir=output_dir,
            use_streaming=False
        )
        
        # Akış tabanlı işleme
        processor_streaming = EnhancedPdfProcessor(
            output_dir=output_dir,
            use_streaming=True
        )
        
        # Standart işleme bellek kullanımı
        standard_result, standard_peak_memory = get_peak_memory_usage(
            processor_standard.process_pdf,
            large_test_pdf,
            category="test"
        )
        
        # Belleği temizle
        del processor_standard
        gc.collect()
        time.sleep(1)
        
        # Akış tabanlı işleme bellek kullanımı
        streaming_result, streaming_peak_memory = get_peak_memory_usage(
            processor_streaming.process_pdf_streaming,
            large_test_pdf,
            category="test"
        )
        
        # Sonuçları logla
        print(f"\nStandart işleme bellek kullanımı: {standard_peak_memory:.2f} MB")
        print(f"Akış tabanlı işleme bellek kullanımı: {streaming_peak_memory:.2f} MB")
        print(f"Bellek kullanımı farkı: {standard_peak_memory - streaming_peak_memory:.2f} MB")
        
        # Akış tabanlı işleme daha az bellek kullanmalı
        assert streaming_peak_memory < standard_peak_memory, \
            "Akış tabanlı işleme daha az bellek kullanmalıydı"
    
    def test_parallel_processing_performance(self, test_pdf_dir, output_dir):
        """Paralel işleme performansını ölçen test"""
        pdf_files = [f for f in os.listdir(test_pdf_dir) if f.lower().endswith('.pdf')]
        if len(pdf_files) < 2:
            pytest.skip("Paralel işleme testi için en az 2 PDF dosyası gerekli.")
        
        # Seri işleme
        processor_serial = EnhancedPdfProcessor(output_dir=output_dir)
        
        start_time = time.time()
        serial_results = processor_serial.process_pdf_directory_streaming(
            directory=test_pdf_dir,
            parallel=False
        )
        serial_time = time.time() - start_time
        
        # Sonuçları temizle
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
        # Paralel işleme
        processor_parallel = EnhancedPdfProcessor(
            output_dir=output_dir,
            max_workers=os.cpu_count() or 2
        )
        
        start_time = time.time()
        parallel_results = processor_parallel.process_pdf_directory_streaming(
            directory=test_pdf_dir,
            parallel=True
        )
        parallel_time = time.time() - start_time
        
        # Sonuçları logla
        print(f"\nSeri işleme süresi: {serial_time:.2f} saniye")
        print(f"Paralel işleme süresi: {parallel_time:.2f} saniye")
        print(f"Hızlanma oranı: {serial_time / parallel_time:.2f}x")
        
        # Paralel işleme daha hızlı olmalı (en az %20 hızlanma)
        assert parallel_time < serial_time * 0.8, \
            f"Paralel işleme yeterince hızlanma sağlamadı: {serial_time / parallel_time:.2f}x"
    
    def test_scaling_with_file_size(self, test_pdf_dir, output_dir):
        """Dosya boyutu ile ölçekleme performansını ölçen test"""
        pdf_files = [f for f in os.listdir(test_pdf_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            pytest.skip("Test PDF'leri bulunamadı.")
        
        # Test için ilk PDF'i kullan
        pdf_path = os.path.join(test_pdf_dir, pdf_files[0])
        
        # Farklı boyutlarda PDF'ler oluştur
        scale_factors = [1, 2, 3]
        test_pdfs = []
        
        for scale in scale_factors:
            # Geçici dosya oluştur
            with tempfile.NamedTemporaryFile(suffix=f'_scale_{scale}.pdf', delete=False) as tmp:
                scaled_pdf_path = tmp.name
            
            # PDF'i ölçeklendir
            try:
                import PyPDF2
                writer = PyPDF2.PdfWriter()
                
                with open(pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    # PDF'i scale faktörü kadar çoğalt
                    for _ in range(scale):
                        for page in reader.pages:
                            writer.add_page(page)
                
                # Ölçeklendirilmiş PDF'i yaz
                with open(scaled_pdf_path, 'wb') as f:
                    writer.write(f)
                
                test_pdfs.append(scaled_pdf_path)
                
            except ImportError:
                pytest.skip("PyPDF2 bulunamadı.")
        
        # Performans ölçümü
        processor = EnhancedPdfProcessor(output_dir=output_dir)
        
        processing_times = []
        file_sizes = []
        
        try:
            for pdf_file in test_pdfs:
                # Dosya boyutu
                file_size = os.path.getsize(pdf_file) / 1024  # KB
                file_sizes.append(file_size)
                
                # İşleme süresi
                start_time = time.time()
                processor.process_pdf(pdf_file)
                end_time = time.time()
                
                processing_time = end_time - start_time
                processing_times.append(processing_time)
                
                print(f"\nDosya boyutu: {file_size:.2f} KB, İşleme süresi: {processing_time:.2f} saniye")
            
            # Ölçekleme analizi
            if len(processing_times) > 2:
                # İşleme süresi, dosya boyutuyla lineer olarak artmalı
                # Basit lineer regresyon ile eğimi hesapla
                x = np.array(file_sizes)
                y = np.array(processing_times)
                
                slope = np.polyfit(x, y, 1)[0]
                
                print(f"Ölçekleme eğimi: {slope:.6f} saniye/KB")
                
                # Sonuçları değerlendir
                assert slope > 0, "İşleme süresi dosya boyutuyla artmalı"
                
                # R-kare (belirlilik katsayısı) hesapla
                y_mean = np.mean(y)
                ss_total = np.sum((y - y_mean) ** 2)
                y_pred = slope * x + np.polyfit(x, y, 1)[1]
                ss_residual = np.sum((y - y_pred) ** 2)
                r_squared = 1 - (ss_residual / ss_total)
                
                print(f"R-kare: {r_squared:.4f}")
                
                # Lineer ölçekleme beklenir (R-kare > 0.8)
                assert r_squared > 0.8, "İşleme süresi dosya boyutuyla lineer ölçeklenmeli"
        
        finally:
            # Geçici dosyaları temizle
            for pdf_file in test_pdfs:
                if os.path.exists(pdf_file):
                    os.unlink(pdf_file)
    
    @pytest.mark.slow
    def test_load_testing(self, test_pdf_dir, output_dir):
        """Yük testi - birden çok PDF'i paralel olarak işleme"""
        pdf_files = [f for f in os.listdir(test_pdf_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            pytest.skip("Test PDF'leri bulunamadı.")
            
        # Çok PDF ile test için dosyaları çoğalt
        test_load_dir = os.path.join(output_dir, "load_test_pdfs")
        os.makedirs(test_load_dir, exist_ok=True)
        
        # Her dosyayı 5 kez kopyala
        for pdf_file in pdf_files:
            for i in range(5):
                src = os.path.join(test_pdf_dir, pdf_file)
                dst = os.path.join(test_load_dir, f"copy_{i}_{pdf_file}")
                shutil.copy(src, dst)
        
        # Paralel işleme
        processor = EnhancedPdfProcessor(
            output_dir=os.path.join(output_dir, "results"),
            max_workers=os.cpu_count() or 2
        )
        
        start_time = time.time()
        results = processor.process_pdf_directory_streaming(
            directory=test_load_dir,
            parallel=True
        )
        total_time = time.time() - start_time
        
        # Toplam işlenen dosya sayısı
        total_processed = len(results)
        
        # Sonuçları logla
        print(f"\nYük testi: {total_processed} PDF dosyası işlendi")
        print(f"Toplam süre: {total_time:.2f} saniye")
        print(f"Dosya başına ortalama süre: {total_time / total_processed:.2f} saniye")
        
        # Tüm dosyalar işlenmiş olmalı
        assert total_processed == len([f for f in os.listdir(test_load_dir) if f.lower().endswith('.pdf')]), \
            "Bazı PDF dosyaları işlenemedi" 