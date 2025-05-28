"""
Model indirme betiği.
Bu betik, kredi RAG sistemi için gerekli modelleri indirir.
"""

import os
import sys
import logging
from pathlib import Path

# Ana dizini ekle
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

# Loglama ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def download_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_dir=None):
    """
    Embedding modelini indirir ve önbelleğe alır.
    
    Args:
        model_name: İndirilecek modelin adı
        cache_dir: Önbellek dizini
    """
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        
        if cache_dir is None:
            cache_dir = os.path.join(ROOT_DIR, "models", "embeddings", ".cache")
            
        # Dizini oluştur
        os.makedirs(cache_dir, exist_ok=True)
        
        # Ortam değişkenlerini ayarla
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir
        
        logger.info(f"'{model_name}' embedding modeli indiriliyor...")
        logger.info(f"Önbellek dizini: {cache_dir}")
        
        # Cihaz tipini belirle
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon için
        else:
            device = "cpu"
            
        logger.info(f"Kullanılacak cihaz: {device}")
        
        # Modeli önce CPU'da indir ve yükle
        model = SentenceTransformer(model_name, cache_folder=cache_dir, device="cpu")
        
        # Ardından gerekirse GPU/MPS'ye taşı
        if device != "cpu":
            try:
                model.to(device)
                logger.info(f"Model {device} cihazına taşındı")
            except Exception as e:
                logger.warning(f"Model {device} cihazına taşınırken hata oluştu, CPU kullanılacak: {str(e)}")
        
        # Test amaçlı basit bir metni kodla
        test_text = "Bu bir test cümlesidir."
        logger.info(f"Test metni kodlanıyor: '{test_text}'")
        _ = model.encode(test_text)
        
        logger.info(f"'{model_name}' modeli başarıyla indirildi ve test edildi.")
        return True
    except Exception as e:
        logger.error(f"Model indirme hatası: {str(e)}")
        
        # Alternatif model deneme
        try:
            backup_model = "sentence-transformers/all-MiniLM-L6-v2"
            logger.warning(f"Alternatif model denenecek: {backup_model}")
            
            model = SentenceTransformer(backup_model, cache_folder=cache_dir, device="cpu")
            test_text = "Bu bir test cümlesidir."
            _ = model.encode(test_text)
            
            logger.info(f"'{backup_model}' alternatif modeli başarıyla indirildi ve test edildi.")
            return True
        except Exception as e2:
            logger.error(f"Alternatif model indirme hatası: {str(e2)}")
            return False

def download_llm_model():
    """
    Mistral LLM modelini indirir.
    """
    try:
        from models.llm import download_mistral_model
        
        model_dir = os.path.join(ROOT_DIR, "models")
        logger.info("Mistral LLM modeli kontrol ediliyor...")
        
        model_path = download_mistral_model(save_dir=model_dir)
        
        logger.info(f"Mistral LLM modeli başarıyla indirildi/kontrol edildi: {model_path}")
        return True
    except Exception as e:
        logger.error(f"Mistral modeli indirme hatası: {str(e)}")
        return False

def main():
    """
    Ana fonksiyon
    """
    logger.info("Model indirme işlemi başlatılıyor...")
    
    # Embedding modelini indir
    if download_embedding_model():
        logger.info("Embedding modeli başarıyla indirildi.")
    else:
        logger.error("Embedding modeli indirilemedi!")
    
    # LLM modelini indir
    if download_llm_model():
        logger.info("LLM modeli başarıyla indirildi.")
    else:
        logger.error("LLM modeli indirilemedi!")
    
    logger.info("Model indirme işlemi tamamlandı.")

if __name__ == "__main__":
    main() 