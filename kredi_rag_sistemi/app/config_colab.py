"""
Google Colab için optimize edilmiş RAG sistemi konfigürasyon ayarları.
Colab Pro Plus aboneliği için optimize edilmiştir.
"""
import os

# Colab ortamı tespiti
IS_COLAB = os.getenv("COLAB_ENV", "0") == "1"
IS_COLAB_PRO_PLUS = os.getenv("COLAB_PRO_PLUS", "1") == "1"  # Pro Plus varsayılan

# Model ayarları - Colab Pro Plus için optimize edilmiş
MODEL_CONFIG = {
    "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "llm_model": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    "llm_file": "mistral-7b-instruct-v0.2.Q4_K_M.gguf" if not IS_COLAB_PRO_PLUS else "mistral-7b-instruct-v0.2.Q5_K_M.gguf",  # Better quality for Pro Plus
    "llm_temperature": 0.7,
    "llm_max_tokens": 1024 if IS_COLAB_PRO_PLUS else 512,  # Full tokens for Pro Plus
    "llm_n_ctx": 4096 if IS_COLAB_PRO_PLUS else 2048,     # Full context for Pro Plus
    "llm_n_gpu_layers": -1 if IS_COLAB_PRO_PLUS else 0,   # GPU acceleration for Pro Plus
    "llm_n_threads": 8 if IS_COLAB_PRO_PLUS else 4        # More threads for Pro Plus
}

# PDF işleme ayarları - Colab Pro Plus için optimize edilmiş
PDF_CONFIG = {
    "chunk_size": 800 if IS_COLAB_PRO_PLUS else 600,       # Larger chunks for Pro Plus
    "chunk_overlap": 150 if IS_COLAB_PRO_PLUS else 100,    # Standard overlap for Pro Plus
    "max_pages": 200 if IS_COLAB_PRO_PLUS else 50,         # More pages for Pro Plus
    "max_file_size_mb": 100 if IS_COLAB_PRO_PLUS else 25,  # Larger files for Pro Plus
    "enable_caching": True,
    "cache_dir": "./cache",
    "memory_threshold_gb": 4.0 if IS_COLAB_PRO_PLUS else 1.0,  # Higher threshold for Pro Plus
    "use_gpu_ocr": True if IS_COLAB_PRO_PLUS else False,   # GPU OCR for Pro Plus
    "parallel_processing": True if IS_COLAB_PRO_PLUS else False,  # Parallel processing for Pro Plus
    "batch_size": 8 if IS_COLAB_PRO_PLUS else 4           # Larger batches for Pro Plus
}

# Vektör veritabanı ayarları - Colab Pro Plus için optimize edilmiş
VECTOR_DB_CONFIG = {
    "use_faiss": True,
    "memory_cleanup_threshold": 200 if IS_COLAB_PRO_PLUS else 50,  # Less frequent cleanup for Pro Plus
    "top_k": 5 if IS_COLAB_PRO_PLUS else 3,                       # More sources for Pro Plus
    "batch_size": 128 if IS_COLAB_PRO_PLUS else 32,               # Larger batches for Pro Plus
    "index_type": "IndexIVFFlat" if IS_COLAB_PRO_PLUS else "IndexFlatIP",  # Better index for Pro Plus
    "nlist": 100 if IS_COLAB_PRO_PLUS else 50,                    # More clusters for Pro Plus
    "use_gpu_index": True if IS_COLAB_PRO_PLUS else False         # GPU indexing for Pro Plus
}

# Kategori eşleştirmeleri
CATEGORY_MAPPING = {
    "mali tablo": "mali_tablo",
    "faaliyet": "faaliyet_raporu", 
    "fiyat": "fiyat_listesi",
    "pegasus": "finansal_bilgiler",
    "kap": "finansal_bilgiler",
    "ccola": "finansal_bilgiler",
    "coca cola": "finansal_bilgiler",
    "finansal": "finansal_bilgiler"
}

# Sistem promptu - Pro Plus için daha detaylı
SYSTEM_PROMPT = """
Sen kredi başvuruları ve finansal belgeler konusunda uzman bir yapay zeka asistanısın. 
Türkçe sorulara Türkçe cevap ver. Verilen belgelerdeki bilgilere dayanarak kapsamlı ve detaylı yanıtlar ver.
Mali tablolar, finansal oranlar, gelir tabloları ve bilanço analizlerinde uzmanlaşmışsın.
Sayısal verileri doğru bir şekilde raporla ve gerektiğinde hesaplamalar yap.
Bilmediğin konularda tahmin yürütme ve her zaman kaynak belgelere atıfta bulun.
"""

# Örnek sorgular - Pro Plus için daha kapsamlı
SAMPLE_QUERIES = [
    "Adel Kalemcilik'in 2024 yılı cirosu ne kadardır ve önceki yılla karşılaştırıldığında nasıl bir artış göstermiştir?",
    "Pegasus'un finansal verileri nelerdir ve hangi finansal göstergeler öne çıkmaktadır?",
    "Mali tablolarda öne çıkan finansal göstergeler nelerdir ve sektör ortalaması ile nasıl karşılaştırılabilir?",
    "CCOLA'nın 2024 finansal durumu nasıl ve likidite oranları hangi seviyelerdedir?",
    "Şirketlerin borç/özkaynak oranları nedir ve finansal risk seviyesi nasıl değerlendirilebilir?",
    "Karlılık oranları ve aktif devir hızları hangi şirketlerde daha iyi performans göstermektedir?"
]

# Colab Pro Plus özel ayarları
COLAB_PRO_PLUS_CONFIG = {
    "use_gpu": True,                     # GPU acceleration enabled
    "max_memory_gb": 52,                 # Pro Plus higher memory limit
    "high_ram_mode": True,               # Use high-RAM runtime
    "enable_ngrok": True,                # Enable tunneling
    "streamlit_port": 8501,
    "auto_cleanup": False,               # Less aggressive cleanup
    "progress_bars": True,               # Show progress
    "log_level": "INFO",
    "cache_embeddings": True,            # Cache embeddings to disk
    "embedding_cache_dir": "./cache/embeddings",
    "use_premium_models": True,          # Enable premium model features
    "enable_advanced_ocr": True,         # Advanced OCR with GPU
    "parallel_pdf_processing": True,     # Process multiple PDFs in parallel
    "advanced_chunking": True,           # Use advanced text chunking
    "semantic_search_enhancement": True, # Enhanced semantic search
    "gpu_batch_size": 256,              # Large GPU batch sizes
    "cpu_workers": 12,                   # More CPU workers
    "memory_mapping": True,              # Use memory mapping for large files
    "prefetch_embeddings": True         # Prefetch embeddings for speed
}

# Colab Free tier ayarları (fallback)
COLAB_FREE_CONFIG = {
    "use_gpu": False,                    # CPU only for free tier
    "max_memory_gb": 12,                 # Free tier memory limit
    "enable_ngrok": True,                # Enable tunneling
    "streamlit_port": 8501,
    "auto_cleanup": True,                # Aggressive cleanup
    "progress_bars": True,               # Show progress
    "log_level": "INFO",
    "cache_embeddings": True,            # Cache embeddings to disk
    "embedding_cache_dir": "./cache/embeddings"
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_handler": True if IS_COLAB_PRO_PLUS else False,  # File logging for Pro Plus
    "console_handler": True,
    "max_log_size_mb": 50 if IS_COLAB_PRO_PLUS else 10,   # Larger logs for Pro Plus
    "log_rotation": True if IS_COLAB_PRO_PLUS else False   # Log rotation for Pro Plus
}

def get_config():
    """Get configuration based on environment and subscription"""
    if IS_COLAB:
        if IS_COLAB_PRO_PLUS:
            print("🚀 Using Google Colab Pro Plus optimized configuration")
            print(f"📊 Memory limit: {COLAB_PRO_PLUS_CONFIG['max_memory_gb']}GB")
            print(f"🔄 Context size: {MODEL_CONFIG['llm_n_ctx']}")
            print(f"📄 Max file size: {PDF_CONFIG['max_file_size_mb']}MB")
            print(f"📑 Max pages: {PDF_CONFIG['max_pages']} pages/PDF")
            print(f"🎯 GPU acceleration: {'✅ Enabled' if COLAB_PRO_PLUS_CONFIG['use_gpu'] else '❌ Disabled'}")
            colab_config = COLAB_PRO_PLUS_CONFIG
        else:
            print("🔧 Using Google Colab Free tier configuration")
            print(f"📊 Memory limit: {COLAB_FREE_CONFIG['max_memory_gb']}GB")
            print(f"🔄 Context size: {MODEL_CONFIG['llm_n_ctx']}")
            print(f"📄 Max file size: {PDF_CONFIG['max_file_size_mb']}MB")
            colab_config = COLAB_FREE_CONFIG
    else:
        print("🖥️ Using standard local configuration")
        colab_config = {}
    
    return {
        "model": MODEL_CONFIG,
        "pdf": PDF_CONFIG,
        "vector_db": VECTOR_DB_CONFIG,
        "categories": CATEGORY_MAPPING,
        "system_prompt": SYSTEM_PROMPT,
        "samples": SAMPLE_QUERIES,
        "colab": colab_config,
        "logging": LOGGING_CONFIG
    } 