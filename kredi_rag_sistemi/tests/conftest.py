"""
Pytest yapılandırma dosyası.
Test çalıştırma için gerekli yapılandırmalar ve ortak fixture'lar.
"""

import os
import sys
import pytest
import tempfile
import logging
from pathlib import Path

# Test dizinini Python modül yoluna ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test grupları (markerlar) tanımla
def pytest_configure(config):
    """Test gruplama markerları tanımla"""
    config.addinivalue_line("markers", "unit: unit tests")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "e2e: end-to-end tests")
    config.addinivalue_line("markers", "performance: performance tests")
    config.addinivalue_line("markers", "slow: slow running tests")

# Test çalıştırma öncesi test PDF'lerinin varlığını kontrol et
@pytest.fixture(scope="session", autouse=True)
def check_test_pdfs():
    """Test PDF dosyalarının varlığını kontrol et ve gerekirse oluştur"""
    test_pdfs_dir = Path(__file__).parent.parent / "test_pdfs"
    if not test_pdfs_dir.exists() or len(list(test_pdfs_dir.glob("*.pdf"))) == 0:
        # test_pdfs dizini yoksa veya içinde PDF yoksa, örnek PDF'ler oluştur
        try:
            print("Test PDF'leri bulunamadı. Örnek PDF'ler oluşturuluyor...")
            from tests.create_test_pdfs import main as create_pdfs
            create_pdfs()
        except Exception as e:
            pytest.skip(f"Test PDF'leri oluşturulamadı: {str(e)}")

# Test loglarını yapılandır
@pytest.fixture(scope="session")
def setup_test_logging():
    """Test loglarını yapılandır"""
    # Test log dizinini oluştur
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Log dosyası yolu
    log_file = log_dir / "test_run.log"
    
    # Logger oluştur
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)
    
    # Dosya handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Konsol handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Handlerleri ekle
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Global test fixtures

@pytest.fixture(scope="session")
def global_temp_dir():
    """Oturum boyunca kullanılacak geçici dizin"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

@pytest.fixture
def temp_directory():
    """Test fonksiyonu için geçici dizin"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

# Performance test özelliklerini yapılandır
def pytest_addoption(parser):
    """Performance testleri için komut satırı seçenekleri ekle"""
    parser.addoption(
        "--run-performance", 
        action="store_true", 
        default=False, 
        help="Performans testlerini çalıştır"
    )

def pytest_collection_modifyitems(config, items):
    """
    Performans testlerini yapılandır:
    1. --run-performance flag'i verilmediyse performans testlerini atla
    2. Yavaş testlere 'slow' markeri ekle
    """
    if not config.getoption("--run-performance"):
        skip_performance = pytest.mark.skip(reason="--run-performance seçeneği verilmedi")
        for item in items:
            if "performance" in item.keywords:
                item.add_marker(skip_performance)
    
    # Yavaş testleri işaretle
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.slow) 