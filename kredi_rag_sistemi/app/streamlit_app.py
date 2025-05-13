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

# Global RAG instance'ı
rag_instance = None

def initialize_rag():
    """RAG sistemini başlatır"""
    global rag_instance
    
    try:
        # Gerekli dizinleri oluştur
        os.makedirs(os.path.join(ROOT_DIR, "data", "processed"), exist_ok=True)
        os.makedirs(os.path.join(ROOT_DIR, "models", "embeddings"), exist_ok=True)
        os.makedirs(os.path.join(ROOT_DIR, "logs"), exist_ok=True)
        
        # Model yollarını belirle
        model_dir = os.path.join(ROOT_DIR, "models")
        vector_db_path = os.path.join(model_dir, "embeddings", "vector_db.pkl")
        
        # Başlatma durumunu göster
        with st.sidebar:
            with st.status("RAG sistemi başlatılıyor...", expanded=True) as status:
                st.write("Mistral modeli kontrol ediliyor...")
                model_path = download_mistral_model(save_dir=model_dir)
                
                st.write("RAG sistemi başlatılıyor...")
                rag_instance = KrediRAG(
                    model_path=model_path,
                    vector_db_path=vector_db_path,
                    top_k=3
                )
                
                # Daha önce işlenmiş veri var mı kontrol et
                if os.path.exists(vector_db_path):
                    st.write("✅ Vektör veritabanı yüklendi")
                else:
                    st.write("⚠️ Vektör veritabanı bulunamadı - PDF'ler henüz işlenmemiş olabilir")
                
                status.update(label="✅ RAG sistemi başarıyla başlatıldı", state="complete")
        
        return rag_instance
    except Exception as e:
        st.error(f"RAG başlatma hatası: {str(e)}")
        logger.error(f"RAG başlatma hatası: {str(e)}")
        return None

# Sidebar
with st.sidebar:
    st.header("Ayarlar")
    
    # PDF Dosyaları Yükleme Bölümü
    st.subheader("PDF İşleme")
    
    # PDF dizini seçimi
    pdf_dir_options = {
        "Varsayılan Test PDF'leri": os.path.join(ROOT_DIR, "test_pdfs"),
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
    if st.button("PDF'leri İşle"):
        if os.path.exists(selected_pdf_dir):
            with st.status("PDF'ler işleniyor...", expanded=True) as status:
                try:
                    # PDF işleyici oluştur
                    processor = PDFProcessor(selected_pdf_dir)
                    st.write(f"📁 {selected_pdf_dir} dizinindeki PDF'ler işleniyor...")
                    
                    st.info("""
                    ℹ️ PDF işleme süresi, belgelerin boyutuna ve içeriklerine bağlı olarak değişebilir.
                    Büyük tablolar içeren PDF'ler için lütfen sabırlı olun. OCR ve tablo tanıma işlemleri zaman alabilir.
                    
                    Türkçe sorgulamak için, soru-cevap alanında Türkçe sorularınızı yazabilirsiniz. 
                    Tablo sorgularında en iyi sonucu almak için, tablodaki sütun adlarını ve belirli verilerle ilgili detaylı sorular sorun.
                    """)
                    
                    documents = processor.process_pdfs()
                    
                    st.write(f"📄 {len(documents)} PDF belgesi başarıyla işlendi")
                    
                    # RAG'a ekle
                    if not rag_instance:
                        st.write("🔄 RAG sistemi başlatılıyor...")
                        rag_instance = initialize_rag()
                    
                    st.write("🔄 Belgeler vektör veritabanına ekleniyor...")
                    rag_instance.ingest_documents(documents)
                    
                    status.update(label="✅ PDF'ler başarıyla işlendi ve RAG'a eklendi", state="complete")
                    
                    # Belge bilgilerini göster
                    if len(documents) > 0:
                        st.write("📊 İşlenen Belgeler:")
                        doc_data = []
                        for i, doc in enumerate(documents):
                            doc_data.append({
                                "No": i+1,
                                "Dosya Adı": doc.get("metadata", {}).get("filename", "Bilinmiyor"),
                                "Sayfa Sayısı": doc.get("metadata", {}).get("page_count", 0),
                                "Kategori": doc.get("metadata", {}).get("category", "Genel")
                            })
                        
                        st.dataframe(pd.DataFrame(doc_data))
                        
                except Exception as e:
                    st.error(f"PDF işleme hatası: {str(e)}")
                    logger.error(f"PDF işleme hatası: {str(e)}")
                    status.update(label="❌ PDF'ler işlenirken hata oluştu", state="error")
        else:
            st.error(f"Dizin bulunamadı: {selected_pdf_dir}")
    
    # RAG bilgisi
    st.divider()
    st.subheader("Sistem Bilgisi")
    
    if rag_instance:
        doc_count = rag_instance.get_document_count()
        st.info(f"📊 Vektör Veritabanında {doc_count} Belge Parçası")
    else:
        st.warning("⚠️ RAG sistemi henüz başlatılmadı")
        if st.button("RAG Sistemini Başlat"):
            initialize_rag()

# Ana içerik alanı
tab1, tab2 = st.tabs(["Soru-Cevap", "Hakkında"])

# Soru-Cevap tab'ı
with tab1:
    # RAG başlatma kontrolü
    if not rag_instance:
        rag_instance = initialize_rag()
    
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
            if not rag_instance:
                st.warning("RAG sistemi başlatılıyor...")
                rag_instance = initialize_rag()
                if not rag_instance:
                    st.error("RAG sistemi başlatılamadı!")
            
            # Sorguyu işle
            with st.spinner(f"'{query}' sorgusu işleniyor..."):
                try:
                    start_time = time.time()
                    result = rag_instance.query(query, top_k=top_k)
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
if not rag_instance:
    rag_instance = initialize_rag()

# RAG instance'ını global olarak saklayan fonksiyonu güncelle
def get_rag_instance(force_recreate_db: bool = False):
    """
    Uygulamanın diğer bölümleri için global bir RAG instance'ı döndürür
    
    Args:
        force_recreate_db: Vektör veritabanını zorla yeniden oluştur
        
    Returns:
        Başlatılmış KrediRAG instance'ı
    """
    global rag_instance
    
    if rag_instance is None:
        rag_instance = initialize_rag()
        
    return rag_instance 