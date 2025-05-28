import os
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging(
    log_dir=None,
    log_level=logging.INFO,
    app_name="kredi_rag",
    console_output=True,
    file_output=True,
    max_file_size=10 * 1024 * 1024,  # 10 MB
    backup_count=5
):
    """
    Sistemin logging yapılandırmasını ayarlar.
    
    Args:
        log_dir: Logların kaydedileceği dizin. Belirtilmezse, 'logs/' klasörü kullanılır.
        log_level: Logging seviyesi (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        app_name: Uygulama adı (log dosya adı için kullanılır)
        console_output: Konsola çıktı verilip verilmeyeceği
        file_output: Dosyaya log kaydedilip kaydedilmeyeceği
        max_file_size: Maksimum log dosyası boyutu (byte cinsinden)
        backup_count: Yedeklenecek log dosyası sayısı
    
    Returns:
        Yapılandırılmış logger nesnesi
    """
    # Proje kök dizini
    root_dir = Path(__file__).parent.parent.absolute()
    
    # Log dizinini ayarla
    if log_dir is None:
        log_dir = os.path.join(root_dir, "logs")
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Ana logger'ı al ve yapılandır
    logger = logging.getLogger("kredi_rag")
    logger.setLevel(log_level)
    logger.handlers = []  # Mevcut handler'ları temizle
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Konsol handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Dosya handler
    if file_output:
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, f"{app_name}.log"),
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Özel istisna yakalama için hook
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Klavye kesintilerinde normal davran
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # Bütün diğer istisnaları logla
        logger.critical("Yakalanmamış istisna:", exc_info=(exc_type, exc_value, exc_traceback))
    
    # Global istisna yakalayıcıyı ayarla
    sys.excepthook = handle_exception
    
    return logger

# Kullanım örneği (doğrudan çağrıldığında)
if __name__ == "__main__":
    logger = setup_logging(log_level=logging.DEBUG)
    logger.debug("Debug mesajı")
    logger.info("Info mesajı")
    logger.warning("Uyarı mesajı")
    logger.error("Hata mesajı")
    logger.critical("Kritik hata mesajı") 