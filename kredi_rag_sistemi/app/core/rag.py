import sys
import os

# Bu satırları geliştirme ortamınıza göre ayarlayın
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.llm import MistralLLM
from models.embeddings import DocumentEmbedder
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class KrediRAG:
    """
    Kredi uygulamaları için Retrieval-Augmented Generation (RAG) sistemini sağlayan ana sınıf.
    Bu sınıf, document embedder ve LLM bileşenlerini birleştirerek tam bir RAG iş akışı sunar.
    """
    
    def __init__(
        self,
        llm: Optional[MistralLLM] = None,
        embedder: Optional[DocumentEmbedder] = None,
        vector_db_path: str = None,
        model_path: str = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 3,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        KrediRAG sınıfının başlatıcısı.
        
        Args:
            llm: Önceden oluşturulmuş MistralLLM örneği (opsiyonel)
            embedder: Önceden oluşturulmuş DocumentEmbedder örneği (opsiyonel)
            vector_db_path: Vektör veritabanı dosya yolu (yoksa oluşturulur)
            model_path: Mistral model dosyasının yolu (llm verilmezse kullanılır)
            embedding_model: Embedding model adı (embedder verilmezse kullanılır)
            top_k: Aramalarda döndürülecek en alakalı belge sayısı
            chunk_size: Belgenin bölüneceği maksimum uzunluk
            chunk_overlap: İki bölüm arasındaki kesişim miktarı
        """
        # LLM başlatma
        self.llm = llm or MistralLLM(model_path=model_path)
        
        # Embedder başlatma
        self.embedder = embedder or DocumentEmbedder(
            model_name=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Vektör veritabanını yükleme (varsa)
        if vector_db_path and os.path.exists(vector_db_path):
            logger.info(f"Vektör veritabanı yükleniyor: {vector_db_path}")
            self.embedder.load_vector_db(vector_db_path)
        
        self.vector_db_path = vector_db_path
        self.top_k = top_k
        
    def ingest_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Belgeleri RAG sistemine ekler"""
        logger.info(f"{len(documents)} belge vektör veritabanına ekleniyor...")
        self.embedder.embed_documents(documents)
        
        if self.vector_db_path:
            self.embedder.save_vector_db(self.vector_db_path)
            logger.info(f"Vektör veritabanı kaydedildi: {self.vector_db_path}")
    
    def query(self, query: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Kullanıcı sorgusunu işler ve yanıt oluşturur"""
        relevant_docs = self.embedder.search_similar(query, self.top_k)
        
        # En alakalı belge parçalarını seç
        context_texts = []
        used_docs = set()
        
        for doc in relevant_docs:
            doc_id = doc["metadata"].get("original_id", doc["id"])
            if doc_id not in used_docs:
                context_texts.append(doc["text"])
                used_docs.add(doc_id)
                if len(context_texts) >= self.top_k:
                    break
        
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
            filepath: Kaydedilecek dosya yolu (belirtilmezse init'te verilen yol kullanılır)
        """
        save_path = filepath or self.vector_db_path
        if save_path:
            self.embedder.save_vector_db(save_path)
            logger.info(f"Vektör veritabanı kaydedildi: {save_path}")
            
    def load_vector_db(self, filepath: Optional[str] = None) -> None:
        """
        Vektör veritabanını diskten yükler.
        
        Args:
            filepath: Yüklenecek dosya yolu (belirtilmezse init'te verilen yol kullanılır)
        """
        load_path = filepath or self.vector_db_path
        if load_path and os.path.exists(load_path):
            self.embedder.load_vector_db(load_path)
            logger.info(f"Vektör veritabanı yüklendi: {load_path}") 