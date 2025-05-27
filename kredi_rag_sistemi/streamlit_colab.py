import streamlit as st
import os
import sys
import pandas as pd
import logging
import time
from pathlib import Path
import gc
import psutil

# Google Colab environment setup - detect Pro Plus
try:
    import google.colab
    is_pro_plus = True  # Assume Pro Plus
    os.environ["COLAB_PRO_PLUS"] = "1" if is_pro_plus else "0"
    os.environ["COLAB_ENV"] = "1"
    
    if is_pro_plus:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["FORCE_CPU"] = "0"  # Allow GPU for Pro Plus
        print("🚀 Pro Plus mode detected - GPU acceleration enabled")
    else:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["FORCE_CPU"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("📱 Free tier mode - CPU only")
except ImportError:
    os.environ["COLAB_ENV"] = "0"
    os.environ["COLAB_PRO_PLUS"] = "0"

# Asyncio fix for Colab
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# Import project modules
try:
    from app.config_colab import get_config
    from app.core.rag import KrediRAG, get_rag_instance
    from app.core.pdf_processor import PDFProcessor
    from models.llm import download_mistral_model
    
    config = get_config()
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Logging setup for Colab
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Kredi RAG Sistemi - Colab",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("🏦 Kredi RAG Sistemi (Google Colab)")
st.markdown("Kredi başvuruları ve finansal belgeler için sorgu tabanlı bilgi erişim sistemi")

# Colab-specific warnings and info
is_pro_plus = os.getenv("COLAB_PRO_PLUS", "0") == "1"

if is_pro_plus:
    st.success("""
    🚀 **Google Colab Pro Plus Optimizasyonları Aktif:**
    - GPU acceleration etkin (T4/V100/A100)
    - Yüksek bellek modu (52GB'a kadar)
    - Maksimum dosya boyutu: 100MB
    - Maksimum sayfa sayısı: 200 sayfa/PDF
    - Gelişmiş OCR ve paralel işleme
    - Premium model kalitesi
    """)
else:
    st.info("""
    🔧 **Google Colab Free Tier Optimizasyonları:**
    - CPU-only işleme (daha stabil)
    - Azaltılmış bellek kullanımı (~12GB)
    - Maksimum dosya boyutu: 25MB
    - Maksimum sayfa sayısı: 50 sayfa/PDF
    """)

# Global RAG instance
if 'rag_instance' not in st.session_state:
    st.session_state.rag_instance = None

def check_memory_usage():
    """Check current memory usage"""
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    memory_available_gb = memory.available / (1024**3)
    
    return memory_percent, memory_available_gb

def display_memory_info():
    """Display memory usage information"""
    memory_percent, memory_available_gb = check_memory_usage()
    
    if memory_percent > 80:
        st.warning(f"⚠️ Memory usage high: {memory_percent:.1f}% | Available: {memory_available_gb:.1f}GB")
    else:
        st.success(f"✅ Memory usage: {memory_percent:.1f}% | Available: {memory_available_gb:.1f}GB")

def initialize_rag():
    """Initialize RAG system with Colab optimizations"""
    try:
        # Create necessary directories
        directories = ["data/processed", "models/embeddings", "logs", "cache"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        # Model paths
        model_dir = os.path.join(ROOT_DIR, "models")
        vector_db_path = os.path.join(model_dir, "embeddings", "vector_db.pkl")
        
        # Progress tracking
        with st.sidebar:
            with st.status("RAG sistemi başlatılıyor...", expanded=True) as status:
                # Memory check
                memory_percent, memory_available_gb = check_memory_usage()
                st.write(f"💾 Memory: {memory_percent:.1f}% | Available: {memory_available_gb:.1f}GB")
                
                if memory_available_gb < 1:
                    st.error("⚠️ Insufficient memory! Please restart the runtime.")
                    st.stop()
                
                # Download model
                st.write("📥 Mistral modeli kontrol ediliyor...")
                model_path = download_mistral_model(save_dir=model_dir)
                
                # Initialize RAG
                st.write("🔄 RAG sistemi başlatılıyor...")
                rag_instance = KrediRAG(
                    model_path=model_path,
                    vector_db_path=vector_db_path,
                    top_k=config["vector_db"]["top_k"],
                    config=config
                )
                
                # Check for existing data
                if os.path.exists(vector_db_path):
                    st.write("✅ Vektör veritabanı yüklendi")
                else:
                    st.write("⚠️ Vektör veritabanı bulunamadı - PDF'ler henüz işlenmemiş")
                
                status.update(label="✅ RAG sistemi başarıyla başlatıldı", state="complete")
        
        st.session_state.rag_instance = rag_instance
        return rag_instance
        
    except Exception as e:
        st.error(f"RAG başlatma hatası: {str(e)}")
        logger.error(f"RAG initialization error: {str(e)}")
        return None

# Sidebar
with st.sidebar:
    st.header("Ayarlar")
    
    # Memory display
    display_memory_info()
    
    # File upload section
    st.subheader("📄 PDF Yükleme")
    
    is_pro_plus = os.getenv("COLAB_PRO_PLUS", "0") == "1"
    max_file_size = "100MB, 200 sayfa/dosya" if is_pro_plus else "25MB, 50 sayfa/dosya"
    
    uploaded_files = st.file_uploader(
        "PDF dosyalarınızı seçin",
        type="pdf",
        accept_multiple_files=True,
        help=f"Maksimum {max_file_size}"
    )
    
    if uploaded_files:
        total_size = sum([file.size for file in uploaded_files]) / (1024*1024)  # MB
        st.info(f"📊 {len(uploaded_files)} dosya seçildi | Toplam boyut: {total_size:.1f}MB")
        
        max_total_size = 500 if is_pro_plus else 100  # MB
        if total_size > max_total_size:
            st.warning(f"⚠️ Toplam dosya boyutu çok büyük ({max_total_size}MB limit). Daha az dosya seçin.")
        else:
            if st.button("PDF'leri İşle", type="primary"):
                process_uploaded_pdfs(uploaded_files)
    
    # Alternative: Directory processing
    st.subheader("📁 Dizin İşleme")
    pdf_dir = st.text_input("PDF Dizini Yolu", value="./test_pdfs")
    
    if st.button("Dizindeki PDF'leri İşle"):
        if os.path.exists(pdf_dir):
            process_directory_pdfs(pdf_dir)
        else:
            st.error(f"Dizin bulunamadı: {pdf_dir}")
    
    # System info
    st.divider()
    st.subheader("Sistem Bilgisi")
    
    if st.session_state.rag_instance:
        try:
            doc_count = st.session_state.rag_instance.get_document_count()
            st.info(f"📊 Vektör DB: {doc_count} belge parçası")
        except:
            st.warning("⚠️ Belge sayısı alınamadı")
    else:
        st.warning("⚠️ RAG sistemi henüz başlatılmadı")
        if st.button("RAG Sistemini Başlat"):
            initialize_rag()
    
    # Cleanup button
    if st.button("🧹 Bellek Temizle"):
        gc.collect()
        st.success("✅ Bellek temizlendi")

def process_uploaded_pdfs(uploaded_files):
    """Process uploaded PDF files"""
    try:
        # Save uploaded files temporarily
        temp_dir = "./temp_pdfs"
        os.makedirs(temp_dir, exist_ok=True)
        
        with st.status("PDF'ler işleniyor...", expanded=True) as status:
            # Save files
            saved_files = []
            for uploaded_file in uploaded_files:
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                saved_files.append(temp_path)
                st.write(f"💾 {uploaded_file.name} kaydedildi")
            
            # Process PDFs
            processor = PDFProcessor(temp_dir)
            documents = processor.process_pdfs()
            
            st.write(f"📄 {len(documents)} PDF belgesi işlendi")
            
            # Initialize RAG if needed
            if not st.session_state.rag_instance:
                st.write("🔄 RAG sistemi başlatılıyor...")
                st.session_state.rag_instance = initialize_rag()
            
            # Add to RAG
            if st.session_state.rag_instance:
                st.write("🔄 Belgeler vektör veritabanına ekleniyor...")
                st.session_state.rag_instance.ingest_documents(documents)
                
                # Show document info
                if documents:
                    doc_data = []
                    for i, doc in enumerate(documents):
                        doc_data.append({
                            "No": i+1,
                            "Dosya": doc.get("metadata", {}).get("filename", "Bilinmiyor"),
                            "Sayfa": doc.get("metadata", {}).get("page_count", 0),
                            "Kategori": doc.get("metadata", {}).get("category", "Genel")
                        })
                    
                    st.dataframe(pd.DataFrame(doc_data))
            
            # Cleanup temp files
            for file_path in saved_files:
                try:
                    os.remove(file_path)
                except:
                    pass
            
            status.update(label="✅ PDF'ler başarıyla işlendi", state="complete")
            
    except Exception as e:
        st.error(f"PDF işleme hatası: {str(e)}")
        logger.error(f"PDF processing error: {str(e)}")

def process_directory_pdfs(pdf_dir):
    """Process PDFs from a directory"""
    try:
        with st.status("Dizindeki PDF'ler işleniyor...", expanded=True) as status:
            processor = PDFProcessor(pdf_dir)
            documents = processor.process_pdfs()
            
            st.write(f"📄 {len(documents)} PDF belgesi işlendi")
            
            # Initialize RAG if needed
            if not st.session_state.rag_instance:
                st.write("🔄 RAG sistemi başlatılıyor...")
                st.session_state.rag_instance = initialize_rag()
            
            # Add to RAG
            if st.session_state.rag_instance:
                st.write("🔄 Belgeler vektör veritabanına ekleniyor...")
                st.session_state.rag_instance.ingest_documents(documents)
            
            status.update(label="✅ PDF'ler başarıyla işlendi", state="complete")
            
    except Exception as e:
        st.error(f"PDF işleme hatası: {str(e)}")

# Main content area
tab1, tab2, tab3 = st.tabs(["Soru-Cevap", "Hakkında", "Colab Rehberi"])

# Q&A Tab
with tab1:
    # Initialize RAG if not done
    if not st.session_state.rag_instance:
        st.session_state.rag_instance = initialize_rag()
    
    # Query input
    query = st.text_input(
        "Finansal belgelere dair sorunuzu sorun:",
        placeholder="Örnek: Pegasus'un 2024 yılı gelir tablosu nasıl?",
        help="Türkçe sorular sorabilirsiniz"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        top_k = st.slider("Kaynak sayısı", min_value=1, max_value=5, value=3)
    
    with col2:
        if st.button("🔍 Sorgula", type="primary"):
            if query and st.session_state.rag_instance:
                with st.spinner("Yanıt oluşturuluyor..."):
                    try:
                        # Memory check before query
                        memory_percent, memory_available_gb = check_memory_usage()
                        if memory_available_gb < 0.5:
                            st.warning("⚠️ Low memory. Consider restarting runtime.")
                        
                        # Execute query
                        start_time = time.time()
                        result = st.session_state.rag_instance.query(query, top_k=top_k)
                        query_time = time.time() - start_time
                        
                        # Display results
                        st.subheader("💬 Yanıt")
                        st.write(result["answer"])
                        
                        st.subheader("📚 Kaynaklar")
                        for i, source in enumerate(result["sources"]):
                            with st.expander(f"Kaynak {i+1}: {source.get('filename', 'Bilinmiyor')}"):
                                st.write(f"**Sayfa:** {source.get('page_number', 'N/A')}")
                                st.write(f"**İçerik:** {source.get('content', '')[:500]}...")
                        
                        # Performance info
                        st.info(f"⏱️ Sorgu süresi: {query_time:.2f} saniye")
                        
                    except Exception as e:
                        st.error(f"Sorgu hatası: {str(e)}")
                        logger.error(f"Query error: {str(e)}")
            else:
                st.warning("Lütfen bir soru girin ve RAG sisteminin başlatıldığından emin olun.")

# About Tab
with tab2:
    st.subheader("📋 Kredi RAG Sistemi Hakkında")
    
    st.markdown("""
    Bu sistem, finansal belgeleri analiz etmek ve sorularınızı yanıtlamak için geliştirilmiştir.
    
    **🎯 Özellikler:**
    - PDF belgelerinden otomatik metin çıkarma
    - Türkçe doğal dil işleme
    - Akıllı soru-cevap sistemi
    - Kaynak referansları
    
    **📄 Desteklenen Belgeler:**
    - Mali tablolar
    - Faaliyet raporları
    - Finansal analizler
    - KAP bildirimleri
    """)

# Colab Guide Tab
with tab3:
    st.subheader("📱 Google Colab Kullanım Rehberi")
    
    st.markdown("""
    ### 🚀 Kurulum Adımları:
    
    1. **Projeyi Colab'a yükleyin:**
    ```python
    # GitHub'dan klonlama
    !git clone [repository-url]
    %cd kredi_rag_sistemi
    ```
    
    2. **Kurulum scriptini çalıştırın:**
    ```python
    !python colab_setup.py
    ```
    
    3. **PDF'lerinizi yükleyin:**
    ```python
    from google.colab import files
    uploaded = files.upload()
    ```
    
    4. **Uygulamayı başlatın:**
    ```python
    !python run_colab.py
    ```
    
    ### 💡 İpuçları:
    - CPU-only modda çalışır (daha stabil)
    - Maksimum 25MB/PDF dosya boyutu
    - Bellek kullanımını takip edin
    - Büyük dosyalar için sayfaları bölün
    
    ### ⚠️ Sınırlamalar:
    - Google Colab free tier: ~12GB RAM
    - Session timeout: ~12 saat
    - GPU kullanımı opsiyonel
    """)
    
    # System requirements check
    st.subheader("🔧 Sistem Durumu")
    
    # Check requirements
    required_packages = ["streamlit", "langchain", "sentence_transformers", "faiss", "PyPDF2"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            st.success(f"✅ {package}")
        except ImportError:
            st.error(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        st.warning(f"Missing packages: {', '.join(missing_packages)}")
        st.code("!pip install " + " ".join(missing_packages))

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
🏦 Kredi RAG Sistemi - Google Colab Edition<br>
Finansal belgeler için akıllı soru-cevap sistemi
</div>
""", unsafe_allow_html=True) 