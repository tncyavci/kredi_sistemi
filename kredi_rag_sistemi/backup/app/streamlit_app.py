import streamlit as st
import os
import sys
import pandas as pd
import logging
from pathlib import Path
from PIL import Image
import time

# Torch'u CPU modunda zorla çalıştır
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["FORCE_CPU"] = "1"

# Asyncio hatasını önlemek için
import nest_asyncio
nest_asyncio.apply()

# Ana dizini ekle
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from app.core.rag import KrediRAG, get_rag_instance
from app.core.pdf_processor import PDFProcessor
from models.llm import download_mistral_model

# Loglama ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(ROOT_DIR, "logs", "streamlit.log"), mode="a")
    ]
)

logger = logging.getLogger(__name__)

# Sayfa ayarları
st.set_page_config(
    page_title="Kredi RAG Sistemi",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Uygulama başlığı
st.title("🏦 Kredi RAG Sistemi")
st.markdown("Kredi başvuruları ve finansal belgeler için sorgu tabanlı bilgi erişim sistemi")

# Session state initialization for RAG instance
if 'rag_instance' not in st.session_state:
    st.session_state.rag_instance = None
    st.session_state.rag_initialized = False

@st.cache_resource
def get_cached_rag_instance():
    """RAG instance'ını cache'le, böylece sadece bir kez yüklenir"""
    try:
        # Gerekli dizinleri oluştur
        os.makedirs(os.path.join(ROOT_DIR, "data", "processed"), exist_ok=True)
        os.makedirs(os.path.join(ROOT_DIR, "models", "embeddings"), exist_ok=True)
        os.makedirs(os.path.join(ROOT_DIR, "logs"), exist_ok=True)
        
        # Model yollarını belirle
        model_dir = os.path.join(ROOT_DIR, "models")
        vector_db_path = os.path.join(ROOT_DIR, "data", "vector_db")
        
        # Mistral modelini kontrol et
        model_path = download_mistral_model(save_dir=model_dir)
        
        # RAG sistemini başlat
        rag_instance = KrediRAG(
            model_path=model_path,
            vector_db_path=vector_db_path,
            top_k=3
        )
        
        logger.info("RAG sistemi cache'e alındı ve başarıyla başlatıldı")
        return rag_instance
        
    except Exception as e:
        logger.error(f"RAG başlatma hatası: {str(e)}")
        st.error(f"RAG başlatma hatası: {str(e)}")
        return None

def initialize_rag():
    """RAG sistemini başlatır - sadece ilk kez çalışır"""
    if not st.session_state.rag_initialized:
        with st.sidebar:
            with st.status("RAG sistemi başlatılıyor...", expanded=True) as status:
                st.write("Cached RAG instance yükleniyor...")
                
                # Cache'den RAG instance'ını al
                st.session_state.rag_instance = get_cached_rag_instance()
                
                if st.session_state.rag_instance:
                    vector_db_path = os.path.join(ROOT_DIR, "data", "vector_db")
                    if os.path.exists(vector_db_path):
                        st.write("✅ Vektör veritabanı mevcut")
                        # Check document count
                        try:
                            doc_count = st.session_state.rag_instance.get_document_count()
                            st.write(f"📊 Veritabanında {doc_count} belge mevcut")
                        except:
                            st.write("📊 Belge sayısı alınamadı")
                    else:
                        st.write("⚠️ Vektör veritabanı bulunamadı - PDF'ler henüz işlenmemiş")
                    
                    st.session_state.rag_initialized = True
                    status.update(label="✅ RAG sistemi başarıyla yüklendi", state="complete")
                else:
                    status.update(label="❌ RAG sistemi yüklenemedi", state="error")
    
    return st.session_state.rag_instance

# Check if RAG needs initialization on startup
if not st.session_state.rag_initialized:
    initialize_rag()

# Sidebar
with st.sidebar:
    st.header("Ayarlar")
    
    # RAG Status Display
    if st.session_state.rag_instance:
        st.success("✅ RAG Sistemi Hazır")
        try:
            doc_count = st.session_state.rag_instance.get_document_count()
            st.info(f"📊 Toplam Belge: {doc_count}")
        except:
            st.info("📊 Belge sayısı hesaplanıyor...")
    else:
        st.error("❌ RAG Sistemi Yüklenemedi")
        if st.button("🔄 RAG'ı Yeniden Başlat"):
            st.session_state.rag_initialized = False
            st.rerun()
    
    st.divider()
    
    # PDF Dosyaları Yükleme Bölümü
    st.subheader("📄 PDF Yükleme")
    
    # Dosya yükleme seçenekleri
    upload_option = st.radio(
        "PDF Yükleme Yöntemi",
        ["Dosya Yükle", "Klasör Seç", "Test PDF'leri"]
    )
    
    documents_to_process = []
    
    if upload_option == "Dosya Yükle":
        # Dosya yükleme widget'ı
        uploaded_files = st.file_uploader(
            "PDF dosyalarınızı seçin",
            type=['pdf'],
            accept_multiple_files=True,
            help="Birden fazla PDF dosyası seçebilirsiniz"
        )
        
        if uploaded_files:
            st.write(f"📁 {len(uploaded_files)} dosya seçildi:")
            for file in uploaded_files:
                st.write(f"• {file.name} ({file.size / 1024:.1f} KB)")
            
            # Yüklenen dosyaları geçici dizine kaydet
            if st.button("Seçilen Dosyaları İşle", type="primary"):
                # Ensure RAG is initialized
                if not st.session_state.rag_instance:
                    st.error("RAG sistemi başlatılamadı. Lütfen sayfayı yenileyin.")
                    st.stop()
                
                temp_dir = os.path.join(ROOT_DIR, "temp_uploads")
                os.makedirs(temp_dir, exist_ok=True)
                
                with st.status("Dosyalar işleniyor...", expanded=True) as status:
                    try:
                        # Dosyaları geçici dizine kaydet
                        saved_files = []
                        for uploaded_file in uploaded_files:
                            file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            saved_files.append(file_path)
                            st.write(f"✅ {uploaded_file.name} kaydedildi")
                        
                        # PDF'leri işle
                        from utils.preprocessing.enhanced_pdf_processor import EnhancedPdfProcessor
                        
                        all_documents = []
                        for file_path in saved_files:
                            st.write(f"🔄 {os.path.basename(file_path)} işleniyor...")
                            
                            # Her dosyayı ayrı ayrı işle
                            file_documents = EnhancedPdfProcessor.process_pdf_to_documents(
                                pdf_path=file_path,
                                category="kullanici_yukleme",
                                source=os.path.basename(file_path),
                                chunk_size=800,
                                overlap=150,
                                use_ocr=True,
                                extract_tables=True
                            )
                            
                            all_documents.extend(file_documents)
                            st.write(f"✅ {len(file_documents)} belge parçası oluşturuldu")
                        
                        # RAG'a ekle
                        st.write("🔄 Belgeler vektör veritabanına ekleniyor...")
                        st.session_state.rag_instance.ingest_documents(all_documents)
                        
                        # Geçici dosyaları temizle
                        for file_path in saved_files:
                            os.remove(file_path)
                        os.rmdir(temp_dir)
                        
                        status.update(label="✅ Dosyalar başarıyla işlendi", state="complete")
                        
                        # İşlenen dosya bilgilerini göster
                        st.success(f"🎉 {len(uploaded_files)} dosya başarıyla işlendi!")
                        st.info(f"📊 Toplam {len(all_documents)} belge parçası oluşturuldu")
                        
                        # Force rerun to update document count
                        time.sleep(1)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Dosya işleme hatası: {str(e)}")
                        logger.error(f"Dosya işleme hatası: {str(e)}")
                        status.update(label="❌ Dosya işleme hatası", state="error")
    
    elif upload_option == "Klasör Seç":
        # Klasör seçimi
        pdf_dir_options = {
            "Proje Dışı PDF Klasörü": os.path.join(os.path.dirname(ROOT_DIR), "pdf"),
            "Özel Dizin": ""
        }
        
        pdf_dir_selection = st.radio("PDF Dizini", list(pdf_dir_options.keys()))
        
        if pdf_dir_selection == "Özel Dizin":
            custom_pdf_dir = st.text_input("PDF Dizini Yolu", "")
            selected_pdf_dir = custom_pdf_dir
        else:
            selected_pdf_dir = pdf_dir_options[pdf_dir_selection]
        
        # PDF işleme butonu
        if st.button("Klasördeki PDF'leri İşle"):
            if os.path.exists(selected_pdf_dir):
                with st.status("PDF'ler işleniyor...", expanded=True) as status:
                    try:
                        # PDF işleyici oluştur
                        processor = PDFProcessor(selected_pdf_dir)
                        st.write(f"📁 {selected_pdf_dir} dizinindeki PDF'ler işleniyor...")
                        
                        documents = processor.process_pdfs()
                        
                        st.write(f"📄 {len(documents)} PDF belgesi başarıyla işlendi")
                        
                        # RAG'a ekle
                        if not st.session_state.rag_instance:
                            st.write("🔄 RAG sistemi başlatılıyor...")
                            st.session_state.rag_instance = initialize_rag()
                        
                        st.write("🔄 Belgeler vektör veritabanına ekleniyor...")
                        st.session_state.rag_instance.ingest_documents(documents)
                        
                        status.update(label="✅ PDF'ler başarıyla işlendi ve RAG'a eklendi", state="complete")
                        
                    except Exception as e:
                        st.error(f"PDF işleme hatası: {str(e)}")
                        logger.error(f"PDF işleme hatası: {str(e)}")
                        status.update(label="❌ PDF'ler işlenirken hata oluştu", state="error")
            else:
                st.error(f"Dizin bulunamadı: {selected_pdf_dir}")
    
    else:  # Test PDF'leri
        selected_pdf_dir = os.path.join(ROOT_DIR, "test_pdfs")
        
        if st.button("Test PDF'lerini İşle"):
            if os.path.exists(selected_pdf_dir):
                with st.status("Test PDF'leri işleniyor...", expanded=True) as status:
                    try:
                        processor = PDFProcessor(selected_pdf_dir)
                        st.write(f"📁 Test PDF'leri işleniyor...")
                        
                        documents = processor.process_pdfs()
                        
                        st.write(f"📄 {len(documents)} test belgesi başarıyla işlendi")
                        
                        # RAG'a ekle
                        if not st.session_state.rag_instance:
                            st.write("🔄 RAG sistemi başlatılıyor...")
                            st.session_state.rag_instance = initialize_rag()
                        
                        st.write("🔄 Belgeler vektör veritabanına ekleniyor...")
                        st.session_state.rag_instance.ingest_documents(documents)
                        
                        status.update(label="✅ Test PDF'leri başarıyla işlendi", state="complete")
                        
                    except Exception as e:
                        st.error(f"PDF işleme hatası: {str(e)}")
                        logger.error(f"PDF işleme hatası: {str(e)}")
                        status.update(label="❌ PDF'ler işlenirken hata oluştu", state="error")
            else:
                st.error(f"Test PDF dizini bulunamadı: {selected_pdf_dir}")

    # System Management Section
    st.divider()
    st.subheader("🔧 Sistem Yönetimi")
    
    # Database clear button
    if st.session_state.rag_instance:
        try:
            doc_count = st.session_state.rag_instance.get_document_count()
            st.metric("Toplam Belge", doc_count)
        except:
            st.info("📊 Belge sayısı hesaplanamadı")
        
        # Clear vector database button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Vektör DB Temizle", help="Tüm belgeleri siler"):
                if st.session_state.get("confirm_clear", False):
                    try:
                        st.session_state.rag_instance.clear_vector_db()
                        st.success("✅ Vektör veritabanı temizlendi!")
                        st.session_state["confirm_clear"] = False
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Temizleme hatası: {str(e)}")
                else:
                    st.session_state["confirm_clear"] = True
                    st.warning("⚠️ Tekrar tıklayarak onaylayın")
        
        with col2:
            if st.button("🔄 Cache Temizle"):
                st.cache_resource.clear()
                st.session_state.rag_initialized = False
                st.session_state.rag_instance = None
                st.success("Cache temizlendi! Sayfa yenileniyor...")
                time.sleep(1)
                st.rerun()
    else:
        st.warning("⚠️ RAG sistemi başlatılmadı")

# Ana içerik alanı
tab1, tab2 = st.tabs(["Soru-Cevap", "Hakkında"])

# Soru-Cevap tab'ı
with tab1:
    # RAG başlatma kontrolü
    if not st.session_state.rag_instance:
        st.session_state.rag_instance = initialize_rag()
    
    # Sorgu giriş alanı
    query = st.text_input("Finansal belgelere dair sorunuzu sorun:", 
                        placeholder="Örneğin: 2021 yılı için toplam aktifler ne kadardır? Pegasus'un özel finansal bilgileri nelerdir?")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        top_k = st.slider("Gösterilecek belge sayısı", min_value=1, max_value=10, value=3)
    
    # Sorgulama butonu
    if st.button("Sorgula", type="primary") or (query and st.session_state.get("last_query") != query):
        if query:
            st.session_state["last_query"] = query
            
            # RAG başlatma kontrolü
            if not st.session_state.rag_instance:
                st.warning("RAG sistemi başlatılıyor...")
                st.session_state.rag_instance = initialize_rag()
                if not st.session_state.rag_instance:
                    st.error("RAG sistemi başlatılamadı!")
            
            # Sorguyu işle
            with st.spinner(f"'{query}' sorgusu işleniyor..."):
                try:
                    start_time = time.time()
                    result = st.session_state.rag_instance.query(query, top_k=top_k)
                    end_time = time.time()
                    
                    # Varsayılan yanıt - önemli bir sorun varsa bunu döndür
                    default_answer = "İlgili bilgiyi bulamadım veya yeterli veri yok. Daha fazla PDF eklemek veya sorguyu değiştirmek ister misiniz?"
                    
                    if result:
                        # Sonucu göster
                        st.markdown("### Yanıt")
                        st.markdown(result.get("answer", default_answer))
                        
                        retrieval_time = result.get("retrieval_time", 0)
                        generation_time = result.get("generation_time", 0)
                        total_time = end_time - start_time
                        
                        # Performans bilgilerini göster
                        with st.expander("Performans bilgileri"):
                            st.markdown(f"""
                            - Vektör arama süresi: {retrieval_time:.2f} saniye
                            - LLM yanıt oluşturma süresi: {generation_time:.2f} saniye
                            - Toplam işlem süresi: {total_time:.2f} saniye
                            """)
                        
                        # Referans belgeleri göster
                        if "source_documents" in result:
                            with st.expander(f"Referans belgeleri ({len(result['source_documents'])})"):
                                for i, doc in enumerate(result["source_documents"]):
                                    col1, col2 = st.columns([1, 3])
                                    with col1:
                                        doc_type = doc.get("metadata", {}).get("type", "bilinmiyor")
                                        doc_source = doc.get("metadata", {}).get("source", "bilinmiyor")
                                        confidence = 1.0 - (doc.get("distance", 0) or 0)
                                        st.markdown(f"**#{i+1}** - {doc_type}")
                                        st.progress(min(1.0, confidence))
                                        st.caption(f"Kaynak: {doc_source}")
                                    
                                    with col2:
                                        doc_text = doc.get("text", "")
                                        # HTML içinde güvenli yolla göster, ters eğik çizgileri düzelt
                                        doc_text_html = doc_text.replace("\n", "<br>")
                                        st.markdown(
                                            f"""<div style="max-height: 200px; overflow-y: auto; border: 1px solid #ddd; padding: 8px; border-radius: 4px; font-size: 0.9em;">
                                            {doc_text_html}
                                            </div>""", 
                                            unsafe_allow_html=True
                                        )
                    else:
                        st.warning(default_answer)
                        
                except Exception as e:
                    logger.error(f"Sorgu hatası: {str(e)}")
                    st.error(f"Sorgu işlenirken bir hata oluştu: {str(e)}")
                    st.info("Vektör veritabanında belge olduğundan emin olun ve tekrar deneyin.")

# Hakkında tab'ı
with tab2:
    st.markdown("""
    ### 📜 Kredi RAG Sistemi Hakkında
    
    Bu uygulama, finansal belgeler ve kredi başvuruları hakkında soru-cevap yapmanıza olanak tanır. 
    Retrieval-Augmented Generation (RAG) teknolojisi kullanarak, PDF belgelerinde bulunan bilgilere 
    doğal dil sorguları ile erişmenizi sağlar.
    
    #### 🔍 Özellikler
    
    - PDF belgelerinden otomatik metin çıkarma ve işleme
    - Vektör tabanlı semantik arama
    - Yerel Mistral 7B dil modeli ile yanıt oluşturma
    - Etkileşimli kullanıcı arayüzü
    
    #### 🛠️ Nasıl Kullanılır
    
    1. **PDF İşleme**: Sol menüdeki "PDF'leri İşle" butonu ile belgeleri sisteme yükleyin
    2. **Soru Sorma**: "Soru-Cevap" sekmesinde sorunuzu yazın ve "Sorgula" butonuna tıklayın
    3. **Sonuçları İnceleme**: Sistem yanıtını ve kullanılan belge kaynaklarını görüntüleyin
    
    #### ⚙️ Teknik Detaylar
    
    - **Vektör Veritabanı**: ChromaDB/FAISS (yerel depolama)
    - **Embedding Modeli**: SentenceTransformers (all-MiniLM-L6-v2)
    - **Dil Modeli**: Mistral 7B (yerel çalışma)
    """)

# Uygulamayı başlattığımızda sistem başlamazsa (if __name__ == "__main__"): uygulanır
if not st.session_state.rag_instance:
    st.session_state.rag_instance = initialize_rag()

# RAG instance'ını global olarak saklayan fonksiyonu güncelle
def get_rag_instance(force_recreate_db: bool = False):
    """
    Uygulamanın diğer bölümleri için global bir RAG instance'ı döndürür
    
    Args:
        force_recreate_db: Vektör veritabanını zorla yeniden oluştur
        
    Returns:
        Başlatılmış KrediRAG instance'ı
    """
    if st.session_state.rag_instance is None:
        st.session_state.rag_instance = initialize_rag()
        
    return st.session_state.rag_instance 