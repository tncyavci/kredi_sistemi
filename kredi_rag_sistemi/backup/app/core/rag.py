import sys
import os
import time
import hashlib
from functools import lru_cache

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
        top_k: int = 2,
        temperature: float = 0.1,
        max_tokens: int = 512,
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
        
        # Initialize query cache
        self._query_cache = {}
        self._cache_max_size = 50  # Maximum number of cached queries
        
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
        
    def _get_query_cache_key(self, query_text: str, top_k: int, use_financial_optimization: bool) -> str:
        """Generate cache key for query"""
        query_data = f"{query_text.lower().strip()}|{top_k}|{use_financial_optimization}"
        return hashlib.md5(query_data.encode()).hexdigest()
    
    def _get_cached_query(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached query result if available and recent"""
        if cache_key in self._query_cache:
            cached_result = self._query_cache[cache_key]
            # Check if cache entry is less than 5 minutes old
            if time.time() - cached_result['timestamp'] < 300:  # 5 minutes
                logger.info("Query result served from cache")
                return cached_result['result']
            else:
                # Remove expired cache entry
                del self._query_cache[cache_key]
        return None
    
    def _cache_query_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache query result"""
        # Implement simple LRU by removing oldest entries if cache is full
        if len(self._query_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = min(self._query_cache.keys(), 
                           key=lambda k: self._query_cache[k]['timestamp'])
            del self._query_cache[oldest_key]
        
        self._query_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
    
    def ingest_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Belgeleri RAG sistemine ekler - optimize edilmiş versiyon
        
        Args:
            documents: Metin ve metadata içeren belge listesi
        """
        if not documents:
            logger.warning("Eklenecek belge bulunamadı.")
            return
            
        logger.info(f"{len(documents)} belge vektör veritabanına ekleniyor...")
        
        # Check for duplicates based on source and page information
        unique_documents = []
        seen_documents = set()
        
        for doc in documents:
            metadata = doc.get('metadata', {})
            doc_key = (
                metadata.get('source', ''),
                metadata.get('page', 0),
                len(doc.get('text', ''))  # Include text length as additional uniqueness check
            )
            
            if doc_key not in seen_documents:
                seen_documents.add(doc_key)
                unique_documents.append(doc)
            else:
                logger.debug(f"Duplicate document detected and skipped: {doc_key}")
        
        if len(unique_documents) != len(documents):
            logger.info(f"Filtered {len(documents) - len(unique_documents)} duplicate documents")
        
        if not unique_documents:
            logger.warning("No unique documents to add after filtering")
            return
        
        # Add unique documents to vector store
        try:
            self.vector_store.add_documents(unique_documents)
            # Clear query cache since new documents might change query results
            self._query_cache.clear()
            logger.info(f"{len(unique_documents)} unique documents successfully added to vector database")
            logger.info("Query cache cleared due to new document additions")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def query(self, query_text: str, top_k: Optional[int] = None, use_financial_optimization: bool = True) -> Dict[str, Any]:
        """
        Bir sorguyu yürüterek ilgili belgeleri alır ve yanıt oluşturur.
        
        Args:
            query_text: Kullanıcı sorgusu
            top_k: Alınacak en alakalı belge sayısı
            use_financial_optimization: Finansal optimizasyon kullanılsın mı
            
        Returns:
            Yanıt ve ilgili bilgileri içeren sözlük
        """
        start_time = time.time()
        
        # top_k için öncelik: parametre > instance değeri > varsayılan
        _top_k = top_k if top_k is not None else self.top_k
        
        try:
            # Check cache
            cache_key = self._get_query_cache_key(query_text, _top_k, use_financial_optimization)
            cached_result = self._get_cached_query(cache_key)
            if cached_result:
                return cached_result
            
            # 1. Vektör veritabanında ilgili belgeleri ara
            retrieval_start = time.time()
            
            # Finansal optimizasyon kullanılacaksa
            if use_financial_optimization and hasattr(self.vector_store, 'query_financial_documents'):
                # Financial query optimization
                financial_filters = self._extract_financial_filters(query_text)
                retrieved_docs = self.vector_store.query_financial_documents(
                    query_text, 
                    top_k=_top_k,
                    financial_filters=financial_filters,
                    prioritize_tables=True
                )
            else:
                # Standard query
                retrieved_docs = self.vector_store.query_documents(query_text, top_k=_top_k)
            
            # Belge sayısını logla
            logger.info(f"Sorgu için bulunan belgeler: {len(retrieved_docs)}")
            
            retrieval_time = time.time() - retrieval_start
            
            # 2. Yanıt oluştur
            generation_start = time.time()
            
            # Enhanced system prompt for financial queries
            system_prompt = self._create_financial_system_prompt(query_text, retrieved_docs)
            
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
                "total_time": time.time() - start_time,
                "financial_optimization_used": use_financial_optimization
            }
            
            logger.info(f"Sorgu başarıyla tamamlandı. Toplam süre: {result['total_time']:.2f} saniye")
            
            # Cache result
            self._cache_query_result(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Sorgu işleme hatası: {str(e)}")
            return {
                "answer": f"Sorgu işlenirken bir hata oluştu: {str(e)}",
                "error": str(e),
                "total_time": time.time() - start_time
            }
    
    def _extract_financial_filters(self, query_text: str) -> Dict[str, Any]:
        """
        Extract financial filters from query text.
        
        Args:
            query_text: User query
            
        Returns:
            Dictionary of financial filters
        """
        filters = {}
        query_lower = query_text.lower()
        
        # Extract years
        import re
        years = re.findall(r'20\d{2}', query_text)
        if years:
            filters['years'] = years[0]  # Use first found year
        
        # Detect financial table types
        if any(term in query_lower for term in ['bilanço', 'balance sheet', 'mali durum']):
            filters['table_type'] = 'balance_sheet'
        elif any(term in query_lower for term in ['gelir tablosu', 'income statement', 'kar zarar']):
            filters['table_type'] = 'income_statement'
        elif any(term in query_lower for term in ['nakit akış', 'cash flow']):
            filters['table_type'] = 'cash_flow_statement'
        
        # Detect if query is specifically about financial data
        financial_keywords = [
            'varlık', 'aktif', 'yükümlülük', 'borç', 'özsermaye', 'gelir', 'gider',
            'kar', 'zarar', 'nakit', 'asset', 'liability', 'equity', 'revenue', 'expense'
        ]
        
        if any(keyword in query_lower for keyword in financial_keywords):
            filters['only_financial'] = True
        
        return filters
    
    def _create_financial_system_prompt(self, query_text: str, documents: List[Dict[str, Any]]) -> str:
        """
        Create enhanced system prompt for financial queries.
        
        Args:
            query_text: User query
            documents: Retrieved documents
            
        Returns:
            Enhanced system prompt
        """
        base_prompt = """Sen Türkçe finansal asistansın. Verilen belgelerdeki bilgileri kullan.
Belgede bilgi yoksa 'Bu bilgi belgelerde mevcut değil' de. Sayıları doğru belirt."""
        
        # Check if we have financial tables
        has_financial_tables = any(
            doc.get('metadata', {}).get('type') == 'table' and 
            doc.get('metadata', {}).get('is_financial', False)
            for doc in documents
        )
        
        if has_financial_tables:
            financial_enhancement = """

ÖZEL TALİMATLAR:
- Tablolardaki sayısal verileri dikkatli analiz et
- Para birimi bilgisini belirt
- Yıl bilgilerini doğru şekilde ayırt et
- Finansal terimleri Türkçe ve İngilizce karşılıklarıyla açıkla
- Tablo verilerini yorumlarken bağlamı da göz önünde bulundur"""
            
            base_prompt += financial_enhancement
        
        # Add year-specific instructions if year is mentioned in query
        import re
        years = re.findall(r'20\d{2}', query_text)
        if years:
            year_instruction = f"\n- Özellikle {', '.join(years)} yılı/yılları ile ilgili verilere odaklan"
            base_prompt += year_instruction
        
        return base_prompt
    
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
        """Vektör veritabanını ve query cache'ini temizler"""
        try:
            self.vector_store.clear()
            # Clear query cache as well
            self._query_cache.clear()
            logger.info("Vektör veritabanı ve query cache temizlendi.")
        except Exception as e:
            logger.error(f"Vektör veritabanı temizlenemedi: {str(e)}")
    
    def clear_query_cache(self) -> None:
        """Query cache'ini temizler"""
        self._query_cache.clear()
        logger.info("Query cache temizlendi.")
    
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