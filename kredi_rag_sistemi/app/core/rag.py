import sys
import os
import time

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
    Kredi RAG (Retrieval-Augmented Generation) Sistemi.
    PDF dosyalarından çıkarılan bilgilerle soru-cevap yapabilen bir sistem.
    """
    
    def __init__(
        self,
        model_path: str,
        vector_db_path: str = "./data/vector_db",
        top_k: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        force_recreate_db: bool = False
    ):
        """
        KrediRAG sistemini başlatır.
        
        Args:
            model_path: LLM model dosyasının yolu
            vector_db_path: Vektör veritabanı dizini
            top_k: Sorgu başına getirilecek en benzer belge sayısı
            temperature: LLM çıktı sıcaklığı (0.0-1.0)
            max_tokens: Maksimum çıktı token sayısı
            force_recreate_db: Vektör DB'yi zorla yeniden oluştur (dikkat: tüm veriler silinir)
        """
        # Gerekli dizinleri oluştur
        Path(vector_db_path).mkdir(parents=True, exist_ok=True)
        
        # Vektör veritabanını başlat
        from models.vector_store import SecureVectorStore
        self.vector_store = SecureVectorStore(
            persist_directory=vector_db_path,
            collection_name="kredi_rag_documents",
            embedding_function_name="sentence-transformers/all-MiniLM-L6-v2",
            force_recreate=force_recreate_db
        )
        logger.info(f"Vektör veritabanı yüklendi: {vector_db_path}")
        
        # LLM modelini yükle
        from models.llm import MistralLLM
        self.llm = MistralLLM(
            model_path=model_path,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.top_k = top_k
        logger.info(f"KrediRAG sistemi başlatıldı.")
        
        # Bu instance'ı global olarak ayarla (API ve UI için)
        set_global_rag_instance(self)
        
    def ingest_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Belgeleri RAG sistemine ekler
        
        Args:
            documents: Metin ve metadata içeren belge listesi
        """
        if not documents:
            logger.warning("Eklenecek belge bulunamadı.")
            return
            
        logger.info(f"{len(documents)} belge vektör veritabanına ekleniyor...")
        
        # Belgeleri doğrudan vektör veritabanına ekle
        # Embeddingler, SecureVectorStore tarafından otomatik olarak oluşturulacak
        self.vector_store.add_documents(documents)
        
        logger.info(f"{len(documents)} belge başarıyla eklendi.")
    
    def query(self, query_text: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Bir sorguyu yürüterek ilgili belgeleri alır ve yanıt oluşturur.
        
        Args:
            query_text: Kullanıcı sorgusu
            top_k: Alınacak en alakalı belge sayısı
            
        Returns:
            Yanıt ve ilgili bilgileri içeren sözlük
        """
        start_time = time.time()
        
        # top_k için öncelik: parametre > instance değeri > varsayılan
        _top_k = top_k if top_k is not None else self.top_k
        
        try:
            # 1. Vektör veritabanında ilgili belgeleri ara
            retrieval_start = time.time()
            
            # Önce tablo içeren belgelerde ara
            filter_tables = {"type": "table"}
            table_docs = self.vector_store.query_documents(query_text, top_k=_top_k, filter_criteria=filter_tables)
            
            # Sonra normal metin belgelerinde ara
            filter_text = {"type": "text_chunk"}
            text_docs = self.vector_store.query_documents(query_text, top_k=_top_k, filter_criteria=filter_text)
            
            # Tabloların ve metin parçalarının sayısını gösteren log ekle
            logger.info(f"Sorgu için bulunan tablolar: {len(table_docs)}, metin parçaları: {len(text_docs)}")
            
            # Tüm belgeleri birleştir (en fazla top_k)
            all_docs = []
            
            # Önceliklendirilmiş tablo ağırlığı
            table_docs_weight = min(int(_top_k * 0.7), len(table_docs))  # Toplamın %70'i kadar tablo
            
            # Önce tabloları ekle (genellikle daha önemli bilgileri içerirler)
            for i, doc in enumerate(table_docs):
                if i < table_docs_weight:
                    doc["metadata"]["priority"] = "high"  # Yüksek öncelik işaretle
                    all_docs.append(doc)
                    
            # Sonra metin belgelerini ekle
            remaining_slots = _top_k - len(all_docs)
            for i, doc in enumerate(text_docs):
                if i < remaining_slots:
                    all_docs.append(doc)
                    
            # top_k'ya göre kırp
            retrieved_docs = all_docs[:_top_k]
            retrieval_time = time.time() - retrieval_start
            
            # 2. Yanıt oluştur
            generation_start = time.time()
            
            # Sistem promptu
            system_prompt = """
            Sen bir finansal ve bankacılık alanında uzmanlaşmış asistansın. 
            Finansal tablolar, kredi bilgileri, mali raporlar ve şirket dokümanları hakkında yanıtlar vermelisin.
            Yalnızca verilen belgelerdeki bilgilere dayanarak cevap ver. Bilmediğin konularda tahmin yürütme.
            Tablolardaki bilgileri düzgün yorumla ve veri hücreleri arasındaki ilişkileri anlamaya çalış.
            Sunduğun bilgilerin doğru olduğundan emin ol ve gerçekleri çarpıtma.
            Eğer belgelerden soruya cevap veremiyorsan, "Bu soruya cevap verecek yeterli bilgi bulamadım" de.
            
            ÖNEMLİ: Türkçe soruları Türkçe olarak cevapla. İngilizce dökümanlarda bilgi bulsan bile cevabını Türkçe olarak ver.
            Tablolardaki veri hücrelerinde bulunan sayıları ve değerleri doğru yorumla.
            """
            
            # Doküman metinlerini al
            context_texts = [doc["text"] for doc in retrieved_docs]
            
            # LLM ile yanıt oluştur
            answer = self.llm.generate_with_context(query_text, context_texts, system_prompt)
            generation_time = time.time() - generation_start
            
            # 3. Sonuç döndür
            result = {
                "answer": answer,
                "source_documents": retrieved_docs,
                "query": query_text,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": time.time() - start_time
            }
            
            logger.info(f"Sorgu başarıyla tamamlandı. Toplam süre: {result['total_time']:.2f} saniye")
            return result
            
        except Exception as e:
            logger.error(f"Sorgu işleme hatası: {str(e)}")
            return {
                "answer": f"Sorgu işlenirken bir hata oluştu: {str(e)}",
                "error": str(e),
                "total_time": time.time() - start_time
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