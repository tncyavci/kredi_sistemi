"""
RAG sistemi için konfigürasyon ayarları.
"""

# Model ayarları
MODEL_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    "llm_file": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    "llm_temperature": 0.7,
    "llm_max_tokens": 1024,
    "llm_n_ctx": 4096,
    "llm_n_gpu_layers": -1
}

# PDF işleme ayarları
PDF_CONFIG = {
    "chunk_size": 800,
    "chunk_overlap": 150,
    "max_pages": None,
    "max_file_size_mb": 50
}

# Vektör veritabanı ayarları
VECTOR_DB_CONFIG = {
    "use_faiss": True,
    "memory_cleanup_threshold": 100,
    "top_k": 3
}

# Kategori eşleştirmeleri
CATEGORY_MAPPING = {
    "mali tablo": "mali_tablo",
    "faaliyet": "faaliyet_raporu",
    "fiyat": "fiyat_listesi",
    "pegasus": "finansal_bilgiler",
    "kap": "finansal_bilgiler"
}

# Sistem promptu
SYSTEM_PROMPT = """
Sen kredi başvuruları, risk değerlendirmesi ve finansal ürünler konusunda uzman bir 
yapay zeka asistanısın. Bilgiyi açık ve anlaşılır bir şekilde ilet. Yalnızca verilen 
belgelerdeki bilgilere dayanarak cevap ver. Bilmediğin konularda tahmin yürütme.
"""

# Örnek sorgular
SAMPLE_QUERIES = [
    "Adel Kalemcilik'in 2024 yılı cirosu ne kadardır?",
    "KuzeyKablo'nun 2023 fiyat listesinde hangi ürünler var?",
    "Pegasus'un finansal verileri nelerdir?",
    "Mali tablolarda öne çıkan finansal göstergeler nelerdir?"
] 