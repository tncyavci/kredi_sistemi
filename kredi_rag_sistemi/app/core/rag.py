import sys
import os

# Bu satırları geliştirme ortamınıza göre ayarlayın
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.llm import MistralLLM
from models.embeddings import DocumentEmbedder
from models.vector_store import SecureVectorStore
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Global RAG instance (Singleton pattern için)
_global_rag_instance = None

def get_rag_instance():
    """
    Global RAG instance'ını döndürür (Singleton pattern)
    """
    return _global_rag_instance

def set_global_rag_instance(instance):
    """
    Global RAG instance'ını ayarlar
    """
    global _global_rag_instance
    _global_rag_instance = instance

class KrediRAG:
    """
    Kredi uygulamaları için Retrieval-Augmented Generation (RAG) sistemini sağlayan ana sınıf.
    Bu sınıf, document embedder ve LLM bileşenlerini birleştirerek tam bir RAG iş akışı sunar.
    """
    
    def __init__(
        self,
        llm: Optional[MistralLLM] = None,
        embedder: Optional[DocumentEmbedder] = None,
        vector_store: Optional[SecureVectorStore] = None,
        vector_db_path: str = None,
        model_path: str = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 3,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        vector_store_type: str = "chroma",  # 'chroma' veya 'faiss'
        encryption_key: Optional[str] = None  # Şifreleme anahtarı
    ):
        """
        KrediRAG sınıfının başlatıcısı.
        
        Args:
            llm: Önceden oluşturulmuş MistralLLM örneği (opsiyonel)
            embedder: Önceden oluşturulmuş DocumentEmbedder örneği (opsiyonel)
            vector_store: Önceden oluşturulmuş SecureVectorStore örneği (opsiyonel)
            vector_db_path: Vektör veritabanı dizini (yoksa oluşturulur)
            model_path: Mistral model dosyasının yolu (llm verilmezse kullanılır)
            embedding_model: Embedding model adı (embedder verilmezse kullanılır)
            top_k: Aramalarda döndürülecek en alakalı belge sayısı
            chunk_size: Belgenin bölüneceği maksimum uzunluk
            chunk_overlap: İki bölüm arasındaki kesişim miktarı
            vector_store_type: Kullanılacak vektör veritabanı tipi ('chroma' veya 'faiss')
            encryption_key: Vektör veritabanı için opsiyonel şifreleme anahtarı
        """
        # LLM başlatma
        self.llm = llm or MistralLLM(model_path=model_path)
        
        # Embedder başlatma
        self.embedder = embedder or DocumentEmbedder(
            model_name=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.top_k = top_k
        self.vector_db_path = vector_db_path

        # Güvenli vektör deposunu başlat
        if vector_store:
            self.vector_store = vector_store
        else:
            if not vector_db_path:
                # Varsayılan yol
                base_dir = Path(__file__).parent.parent.parent
                vector_db_path = str(base_dir / "data" / "vector_db")
                
            # Güvenli vektör veritabanını başlat
            self.vector_store = SecureVectorStore(
                persist_directory=vector_db_path,
                collection_name="kredi_rag_documents",
                embedding_function_name=embedding_model,
                store_type=vector_store_type,
                encryption_key=encryption_key
            )
            
            # Mevcut vektör veritabanını yükle (varsa)
            try:
                self.vector_store.load()
                logger.info(f"Vektör veritabanı yüklendi: {vector_db_path}")
            except Exception as e:
                logger.warning(f"Vektör veritabanı yüklenemedi (yeni oluşturulacak): {str(e)}")
        
        # Bu instance'ı global olarak ayarla (API ve UI için)
        set_global_rag_instance(self)
        
    def ingest_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Belgeleri RAG sistemine ekler
        
        Args:
            documents: Metin ve metadata içeren belge listesi
        """
        logger.info(f"{len(documents)} belge vektör veritabanına ekleniyor...")
        
        # Belgeler için embeddingler oluştur
        processed_documents = []
        
        for doc in documents:
            # Belge metnini alın
            text = doc.get("text", "")
            if not text:
                continue
                
            # Belgeyi parçalara böl
            chunks = self.embedder._chunk_text(text)
            
            # Her parça için bir belge oluştur
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc.get('id', 'doc')}_{i}"
                
                # Embedding oluştur
                embedding = self.embedder.embed_text(chunk)
                
                # İşlenmiş belgeyi ekle
                processed_documents.append({
                    "id": chunk_id,
                    "text": chunk,
                    "embedding": embedding,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "chunk_index": i,
                        "original_id": doc.get("id", "doc")
                    }
                })
        
        # Belgeleri güvenli vektör veritabanına ekle
        self.vector_store.add_documents(processed_documents)
        
        # Vektör veritabanını kaydet
        self.save_vector_db()
        
        logger.info(f"{len(processed_documents)} belge başarıyla eklendi.")
    
    def query(self, query: str, system_prompt: Optional[str] = None, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Kullanıcı sorgusunu işler ve yanıt oluşturur
        
        Args:
            query: Kullanıcı sorgusu
            system_prompt: Sistem promptu (isteğe bağlı)
            top_k: Döndürülecek belge sayısı (belirtilmezse self.top_k kullanılır)
            
        Returns:
            Yanıt ve ilgili belgeler içeren sözlük
        """
        # Kullanılacak top_k değerini belirle
        actual_top_k = top_k if top_k is not None else self.top_k
        
        # Sorgu için embedding oluştur
        query_embedding = self.embedder.embed_text(query)
        
        # Benzer belgeleri ara
        relevant_docs = self.vector_store.search(query_embedding, actual_top_k)
        
        # En alakalı belge metinlerini seç
        context_texts = [doc["text"] for doc in relevant_docs]
        
        default_system_prompt = """
        Sen kredi başvuruları, risk değerlendirmesi ve finansal ürünler konusunda uzman bir 
        yapay zeka asistanısın. Bilgiyi açık ve anlaşılır bir şekilde ilet. Yalnızca verilen 
        belgelerdeki bilgilere dayanarak cevap ver. Bilmediğin konularda tahmin yürütme.
        """
        
        system_prompt = system_prompt or default_system_prompt
        response = self.llm.generate_with_context(query, context_texts, system_prompt)
        
        return {
            "query": query,
            "response": response,
            "relevant_documents": relevant_docs
        }
        
    def save_vector_db(self, filepath: Optional[str] = None) -> None:
        """
        Vektör veritabanını diske kaydeder.
        
        Args:
            filepath: Kaydedilecek dizin (belirtilmezse init'te verilen yol kullanılır)
        """
        try:
            # Vektör deposunu kaydet
            self.vector_store.save()
            logger.info("Vektör veritabanı başarıyla kaydedildi.")
        except Exception as e:
            logger.error(f"Vektör veritabanı kaydedilemedi: {str(e)}")
            
    def load_vector_db(self, filepath: Optional[str] = None) -> None:
        """
        Vektör veritabanını diskten yükler.
        
        Args:
            filepath: Yüklenecek dizin (belirtilmezse init'te verilen yol kullanılır)
        """
        try:
            # Vektör deposunu yükle
            self.vector_store.load()
            logger.info("Vektör veritabanı başarıyla yüklendi.")
        except Exception as e:
            logger.error(f"Vektör veritabanı yüklenemedi: {str(e)}")
    
    def clear_vector_db(self) -> None:
        """Vektör veritabanını temizler"""
        try:
            self.vector_store.clear()
            logger.info("Vektör veritabanı temizlendi.")
        except Exception as e:
            logger.error(f"Vektör veritabanı temizlenemedi: {str(e)}")
    
    def get_document_count(self) -> int:
        """
        Vektör veritabanındaki belge sayısını döndürür
        
        Returns:
            Belge sayısı
        """
        try:
            return self.vector_store.get_document_count()
        except Exception as e:
            logger.error(f"Belge sayısı alınamadı: {str(e)}")
            return 0 