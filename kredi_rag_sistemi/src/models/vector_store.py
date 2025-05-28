"""
Vektör veritabanı entegrasyonları için modül.
Bu modül, farklı vektör veritabanı çözümlerini destekleyen bir arayüz sağlar.
Güvenlik ve yerel çalışma göz önünde bulundurularak ChromaDB ve gelişmiş FAISS entegrasyonları içerir.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import faiss
import pickle
import hashlib
import json
import time
import base64
import shutil
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class SecureVectorStore:
    """
    Güvenlik odaklı vektör depolama ve arama işlevlerini sağlayan sınıf.
    ChromaDB veya gelişmiş FAISS kullanarak yerel ve güvenli vektör depolama sağlar.
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/vector_db",
        collection_name: str = "kredi_rag_documents",
        embedding_function_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        store_type: str = "chroma",  # 'chroma' veya 'faiss'
        encryption_key: Optional[str] = None,
        force_recreate: bool = False
    ):
        """
        SecureVectorStore sınıfının başlatıcısı.
        
        Args:
            persist_directory: Vektör veritabanının kaydedileceği dizin
            collection_name: ChromaDB koleksiyon adı
            embedding_function_name: Kullanılacak embedding modeli
            store_type: Kullanılacak vektör veritabanı tipi ('chroma' veya 'faiss')
            encryption_key: Opsiyonel şifreleme anahtarı (güvenlik için)
            force_recreate: Varolan koleksiyon zorla yeniden oluşturulsun mu
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function_name = embedding_function_name
        self.store_type = store_type
        self.encryption_key = encryption_key
        self.force_recreate = force_recreate
        
        # Güvenlik kontrolleri
        self._validate_and_create_directory(persist_directory)
        
        if self.store_type == "chroma":
            self._setup_chroma()
        elif self.store_type == "faiss":
            self.faiss_index = None
            self.faiss_id_to_document = {}
            self.faiss_document_embedding_map = {}
            self.embedding_dim = None
        else:
            raise ValueError(f"Desteklenmeyen vektör store tipi: {store_type}")
    
    def _validate_and_create_directory(self, directory: str) -> None:
        """Dizinin güvenli olduğunu doğrular ve gerekirse oluşturur"""
        try:
            # Path injection saldırılarına karşı koruma
            directory_path = Path(directory).resolve()
            directory_str = str(directory_path)
            
            # Dizinin güvenlik kontrolü
            if ".." in directory_str or "~" in directory_str:
                raise ValueError(f"Güvenlik riski: Dizin yolu güvenli değil: {directory}")
            
            # Dizini oluştur
            os.makedirs(directory_path, exist_ok=True)
            logger.info(f"Vektör veritabanı dizini doğrulandı: {directory_path}")
        except Exception as e:
            logger.error(f"Dizin doğrulama hatası: {str(e)}")
            raise
    
    def _setup_chroma(self) -> None:
        """ChromaDB istemcisini ve koleksiyonunu kurar"""
        try:
            # Yeni ChromaDB yapılandırması - deprecation uyarılarını gidermek için
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False  # Telemetri devre dışı (güvenlik için)
                )
            )
            
            # ÇÖZÜM: Hiç embedding function kullanmayacağız, sadece metadata ve text saklayacağız
            # Manuel embedding yapıp direkt vektörleri vereceğiz
            logger.info("ChromaDB embedding fonksiyonu devre dışı - manuel embedding kullanılacak")
            
            # Manuel embedding için SentenceTransformer
            self._init_manual_embedder()
            
            # Koleksiyonu oluştur veya yükle (embedding function olmadan)
            self._create_or_load_collection()
            
            logger.info(f"ChromaDB koleksiyonu başarıyla kuruldu: {self.collection_name}")
        except Exception as e:
            logger.error(f"ChromaDB kurulum hatası: {str(e)}")
            raise
    
    def _init_manual_embedder(self):
        """Manuel embedding için SentenceTransformer başlat"""
        try:
            from sentence_transformers import SentenceTransformer
            import os
            import torch
            import multiprocessing as mp
            
            # Multiprocessing ayarları
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass
            
            # Environment setup
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            
            # Model cache
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   "models", "embeddings", ".cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            logger.info("Manuel SentenceTransformer yükleniyor...")
            self.manual_embedder = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu",
                cache_folder=cache_dir
            )
            logger.info("Manuel SentenceTransformer başarıyla yüklendi")
            
        except Exception as e:
            logger.error(f"Manuel embedder başlatma hatası: {str(e)}")
            raise
    
    def _get_manual_embeddings(self, texts):
        """Manuel embedding hesapla - tek tek işleyerek multiprocessing sorununu önle"""
        import torch
        import gc
        import os
        
        try:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            embeddings = []
            for text in texts:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                with torch.no_grad():
                    try:
                        embedding = self.manual_embedder.encode(
                            text,
                            convert_to_numpy=True,
                            show_progress_bar=False,
                            device='cpu',
                            normalize_embeddings=True
                        )
                        embeddings.append(embedding.tolist())
                    except Exception as e:
                        logger.warning(f"Manuel embedding hatası, sıfır vektör kullanılıyor: {str(e)}")
                        embeddings.append([0.0] * 384)  # all-MiniLM-L6-v2 boyutu
                
                gc.collect()
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Manuel embedding hatası: {str(e)}")
            # Fallback: sıfır vektörler
            return [[0.0] * 384 for _ in texts]
    
    def _encrypt_data(self, data: Any) -> bytes:
        """Veriyi şifreler (opsiyonel güvenlik özelliği)"""
        if not self.encryption_key:
            return pickle.dumps(data)
        
        # Basit şifreleme - üretim ortamında daha gelişmiş bir şifreleme kullanılmalıdır
        from cryptography.fernet import Fernet
        key = hashlib.sha256(self.encryption_key.encode()).digest()
        f = Fernet(base64.urlsafe_b64encode(key[:32]))
        return f.encrypt(pickle.dumps(data))
    
    def _decrypt_data(self, encrypted_data: bytes) -> Any:
        """Şifreli veriyi çözer (opsiyonel güvenlik özelliği)"""
        if not self.encryption_key:
            return pickle.loads(encrypted_data)
        
        # Basit şifre çözme
        from cryptography.fernet import Fernet
        key = hashlib.sha256(self.encryption_key.encode()).digest()
        f = Fernet(base64.urlsafe_b64encode(key[:32]))
        return pickle.loads(f.decrypt(encrypted_data))
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Belgeleri vektör veritabanına ekler.
        
        Args:
            documents: Belge listesi. Her belge {"id", "text", "embedding", "metadata"} içermeli.
        """
        if not documents:
            logger.warning("Eklenmek üzere belge verilmedi.")
            return
        
        try:
            if self.store_type == "chroma":
                self._add_documents_to_chroma(documents)
            elif self.store_type == "faiss":
                self._add_documents_to_faiss(documents)
                
            logger.info(f"{len(documents)} belge vektör veritabanına eklendi.")
        except Exception as e:
            logger.error(f"Belge ekleme hatası: {str(e)}")
            raise
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        ChromaDB için metadata değerlerini temizler.
        ChromaDB sadece str, int, float değerlerini kabul eder.
        """
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                sanitized[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = str(value) if isinstance(value, bool) else value
            elif isinstance(value, (list, tuple)):
                # List/tuple'ı string'e çevir
                sanitized[key] = str(value)
            elif isinstance(value, dict):
                # Dict'i JSON string'e çevir
                import json
                try:
                    sanitized[key] = json.dumps(value)
                except:
                    sanitized[key] = str(value)
            else:
                # Diğer tipleri string'e çevir
                sanitized[key] = str(value)
        return sanitized

    def _add_documents_to_chroma(self, documents: List[Dict[str, Any]]) -> None:
        """Belgeleri ChromaDB'ye ekler - manuel embedding ile"""
        # ChromaDB'nin işleyebileceği maksimum batch boyutu (güvenli değer)
        max_batch_size = 100  # Reduced for manual embedding
        
        # Belgeleri daha küçük batch'lere böl
        for i in range(0, len(documents), max_batch_size):
            batch = documents[i:i + max_batch_size]
            
            ids = [doc["id"] for doc in batch]
            texts = [doc["text"] for doc in batch]
            # Metadata'ları temizle
            metadatas = [self._sanitize_metadata(doc.get("metadata", {})) for doc in batch]
            
            # Manuel embedding hesapla
            logger.info(f"Manuel embedding hesaplanıyor: {len(batch)} belge...")
            manual_embeddings = self._get_manual_embeddings(texts)
            
            # ChromaDB'ye ekle - her zaman manuel embeddingleri kullan
            try:
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    embeddings=manual_embeddings,
                    metadatas=metadatas
                )
                
                logger.info(f"ChromaDB'ye {len(batch)} belge eklendi. (Toplam batch: {i//max_batch_size + 1})")
            except Exception as e:
                logger.error(f"ChromaDB belge ekleme hatası (batch {i//max_batch_size + 1}): {str(e)}")
                # Tekil ekleme stratejisine geç
                logger.info("Tekil belge ekleme stratejisine geçiliyor...")
                for j, (doc_id, text, metadata, embedding) in enumerate(zip(ids, texts, metadatas, manual_embeddings)):
                    try:
                        self.collection.add(
                            ids=[doc_id],
                            documents=[text],
                            embeddings=[embedding],
                            metadatas=[metadata]
                        )
                    except Exception as e2:
                        logger.error(f"Tekil belge ekleme hatası (batch {i//max_batch_size + 1}, doc {j}): {str(e2)}")
                        continue
    
    def _add_documents_to_faiss(self, documents: List[Dict[str, Any]]) -> None:
        """Belgeleri FAISS'e ekler"""
        for doc in documents:
            doc_id = doc["id"]
            text = doc["text"]
            metadata = doc.get("metadata", {})
            
            # Embedding'i al veya hesapla
            if "embedding" in doc and doc["embedding"] is not None:
                embedding = doc["embedding"]
            else:
                # Embedding fonksiyonu yoksa hata ver
                raise ValueError("FAISS için embedding sağlanmalıdır")
            
            # FAISS indeksi başlat
            if self.faiss_index is None:
                self.embedding_dim = len(embedding)
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            
            # Veriyi normalize et
            embedding_np = np.array([embedding]).astype('float32')
            faiss.normalize_L2(embedding_np)
            
            # FAISS'e ekle
            self.faiss_index.add(embedding_np)
            
            # Doküman bilgisini sakla
            self.faiss_id_to_document[self.faiss_index.ntotal - 1] = {
                "id": doc_id,
                "text": text,
                "metadata": metadata
            }
            self.faiss_document_embedding_map[doc_id] = embedding
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Vektör veritabanında sorgu yapar.
        
        Args:
            query_embedding: Sorgunun embedding vektörü.
            top_k: Döndürülecek en benzer belge sayısı.
            
        Returns:
            En benzer belgelerin listesi.
        """
        try:
            if self.store_type == "chroma":
                return self._search_chroma(query_embedding, top_k)
            elif self.store_type == "faiss":
                return self._search_faiss(query_embedding, top_k)
        except Exception as e:
            logger.error(f"Arama hatası: {str(e)}")
            raise
    
    def _search_chroma(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """ChromaDB'de arama yapar"""
        # ChromaDB sonuçlarını al
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Sonuçları formatlayarak döndür
        documents = []
        for i in range(len(results["ids"][0])):
            documents.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1.0 - float(results["distances"][0][i])  # Uzaklığı benzerliğe çevir
            })
        
        return documents
    
    def _search_faiss(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """FAISS'te arama yapar"""
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []
        
        # Sorgu vektörünü hazırla
        query_embedding_np = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_embedding_np)
        
        # FAISS araması yap
        distances, indices = self.faiss_index.search(query_embedding_np, top_k)
        
        # Sonuçları formatlayarak döndür
        documents = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Geçerli indeks
                doc_data = self.faiss_id_to_document[idx]
                documents.append({
                    "id": doc_data["id"],
                    "text": doc_data["text"],
                    "metadata": doc_data["metadata"],
                    "score": float(1.0 - distances[0][i])  # Uzaklığı benzerliğe çevir
                })
        
        return documents
    
    def save(self) -> None:
        """Vektör veritabanını diske kaydeder"""
        try:
            if self.store_type == "chroma":
                # ChromaDB artık PersistentClient kullanılıyor ve otomatik olarak kaydediyor
                # persist() çağrısına artık gerek yok
                logger.info(f"ChromaDB veritabanı diske kaydedildi: {self.persist_directory}")
            elif self.store_type == "faiss":
                # FAISS indeksini ve dokuman eşlemelerini kaydet
                faiss_dir = os.path.join(self.persist_directory, "faiss")
                os.makedirs(faiss_dir, exist_ok=True)
                
                # FAISS indeksini kaydet
                faiss_index_path = os.path.join(faiss_dir, "faiss.index")
                faiss.write_index(self.faiss_index, faiss_index_path)
                
                # Doküman bilgilerini kaydet
                docs_path = os.path.join(faiss_dir, "documents.secure")
                with open(docs_path, "wb") as f:
                    f.write(self._encrypt_data({
                        "faiss_id_to_document": self.faiss_id_to_document,
                        "faiss_document_embedding_map": self.faiss_document_embedding_map,
                        "embedding_dim": self.embedding_dim
                    }))
                
                logger.info(f"FAISS veritabanı diske kaydedildi: {faiss_dir}")
        except Exception as e:
            logger.error(f"Vektör veritabanı kaydetme hatası: {str(e)}")
            raise
    
    def load(self) -> None:
        """Vektör veritabanını diskten yükler"""
        try:
            if self.store_type == "chroma":
                # ChromaDB zaten otomatik olarak persist_directory'den yüklüyor
                logger.info(f"ChromaDB veritabanı diskten yüklendi: {self.persist_directory}")
            elif self.store_type == "faiss":
                faiss_dir = os.path.join(self.persist_directory, "faiss")
                faiss_index_path = os.path.join(faiss_dir, "faiss.index")
                docs_path = os.path.join(faiss_dir, "documents.secure")
                
                if os.path.exists(faiss_index_path) and os.path.exists(docs_path):
                    # FAISS indeksini yükle
                    self.faiss_index = faiss.read_index(faiss_index_path)
                    
                    # Doküman bilgilerini yükle
                    with open(docs_path, "rb") as f:
                        data = self._decrypt_data(f.read())
                        self.faiss_id_to_document = data["faiss_id_to_document"]
                        self.faiss_document_embedding_map = data["faiss_document_embedding_map"]
                        self.embedding_dim = data["embedding_dim"]
                    
                    logger.info(f"FAISS veritabanı diskten yüklendi: {faiss_dir}")
                else:
                    logger.warning(f"FAISS veritabanı bulunamadı: {faiss_dir}")
        except Exception as e:
            logger.error(f"Vektör veritabanı yükleme hatası: {str(e)}")
            raise
    
    def clear(self) -> None:
        """Vektör veritabanını temizler"""
        try:
            if self.store_type == "chroma":
                self.client.delete_collection(self.collection_name)
                self._setup_chroma()
            elif self.store_type == "faiss":
                self.faiss_index = None
                self.faiss_id_to_document = {}
                self.faiss_document_embedding_map = {}
            logger.info("Vektör veritabanı temizlendi.")
        except Exception as e:
            logger.error(f"Veritabanı temizleme hatası: {str(e)}")
            raise

    def get_document_count(self) -> int:
        """
        Vektör veritabanındaki toplam belge sayısını döndürür.
        
        Returns:
            Belge sayısı (int)
        """
        try:
            if self.store_type == "chroma":
                try:
                    # Koleksiyon bilgisini al
                    return self.collection.count()
                except Exception as e:
                    logger.error(f"ChromaDB belge sayısı alınamadı: {str(e)}")
                    return 0
            elif self.store_type == "faiss":
                # FAISS için belge sayısı
                if self.faiss_index is not None:
                    return self.faiss_index.ntotal
                return 0
            return 0
        except Exception as e:
            logger.error(f"Belge sayısı alınamadı: {str(e)}")
            return 0 

    def _create_or_load_collection(self):
        """Koleksiyon oluştur veya varsa yükle - embedding function olmadan"""
        try:
            # Önce koleksiyonun var olup olmadığını kontrol et
            collections = self.client.list_collections()
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if collection_exists and not self.force_recreate:
                # Varolan koleksiyonu yükle - embedding function olmadan
                logger.info(f"Varolan ChromaDB koleksiyonu kullanılıyor: {self.collection_name}")
                self.collection = self.client.get_collection(name=self.collection_name)
            else:
                if collection_exists and self.force_recreate:
                    # Koleksiyonu sil (zorla yeniden oluşturulacaksa)
                    logger.warning(f"Mevcut koleksiyon bulundu ve silinecek: {self.collection_name}")
                    self.client.delete_collection(name=self.collection_name)
                    logger.info(f"Eski koleksiyon silindi: {self.collection_name}")
                
                # Yeni koleksiyon oluştur - embedding function olmadan
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Kredi RAG sistemi için vektör veritabanı - manuel embedding"}
                )
                logger.info(f"Yeni ChromaDB koleksiyonu oluşturuldu (manuel embedding): {self.collection_name}")
            
            logger.info(f"ChromaDB koleksiyonu başarıyla kuruldu: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Koleksiyon oluşturulurken hata: {str(e)}")
            raise
            
        logger.info(f"ChromaDB veritabanı diskten yüklendi: {self.persist_directory}")
        
    def query_documents(
        self, 
        query_text: str, 
        top_k: int = 3,
        filter_criteria: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Vektör veritabanında sorgu yapar - manuel embedding ile
        
        Args:
            query_text: Sorgu metni
            top_k: Döndürülecek en benzer belge sayısı
            filter_criteria: Filtre kriterleri
            
        Returns:
            En benzer belgelerin listesi
        """
        try:
            # Türkçe karakterler için normalizasyon
            import unicodedata
            query_text = unicodedata.normalize('NFKC', query_text)
            
            # Daha fazla sonuç getir sonra filtrele (daha iyi sonuçlar için)
            expanded_top_k = min(top_k * 5, 50)  # Daha fazla sonuç al
            
            # Manuel embedding hesapla
            logger.info(f"Sorgu için manuel embedding hesaplanıyor: {query_text[:50]}...")
            query_embeddings = self._get_manual_embeddings([query_text])
            
            # Sorguyu yap - manuel embedding ile
            query_results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=expanded_top_k,
                where=filter_criteria,
                include=["documents", "metadatas", "distances"]
            )
            
            # Sonuçları formatlı hale getir
            results = []
            if query_results and len(query_results['ids'][0]) > 0:
                for i in range(len(query_results['ids'][0])):
                    # Sonuç kalitesini kontrol et
                    distance = query_results['distances'][0][i] if 'distances' in query_results else 0
                    
                    # Much more permissive threshold for table data
                    # Tables often have different text structure causing higher distances
                    if distance > 3.0 and top_k > 1:  # Increased from 2.5 to 3.0
                        continue
                        
                    result = {
                        "id": query_results['ids'][0][i],
                        "text": query_results['documents'][0][i],
                        "metadata": query_results['metadatas'][0][i],
                        "distance": distance
                    }
                    results.append(result)
            
            # Sort results by relevance with table priority
            def score_result(doc):
                metadata = doc.get('metadata', {})
                base_score = 1.0 / (1.0 + doc['distance'])  # Higher score for lower distance
                
                # Boost table documents
                if metadata.get('type') == 'table':
                    base_score *= 1.5
                    
                # Boost financial documents
                if metadata.get('is_financial', False):
                    base_score *= 1.2
                    
                # Boost documents with financial terms
                financial_terms = ['aktif', 'varlık', 'yükümlülük', 'özsermaye', 'gelir', 'gider', 
                                 'bilanço', 'kar', 'zarar', 'nakit', 'toplam', 'tl', 'asset',
                                 'liability', 'equity', 'revenue', 'income', 'pegasus']
                
                text_lower = doc['text'].lower()
                query_lower = query_text.lower()
                
                # Check for exact financial term matches
                for term in financial_terms:
                    if term in query_lower and term in text_lower:
                        base_score *= 1.1
                        
                return base_score
            
            # Sort by relevance score (descending)
            results.sort(key=score_result, reverse=True)
            
            # Return top_k results
            final_results = results[:top_k]
            
            logger.info(f"Sorgu başarıyla çalıştırıldı, {len(final_results)} sonuç döndürüldü")
            return final_results
            
        except Exception as e:
            logger.error(f"Sorgu sırasında hata: {str(e)}")
            return []
    
    def query_financial_documents(
        self, 
        query_text: str, 
        top_k: int = 3,
        financial_filters: Optional[Dict[str, Any]] = None,
        prioritize_tables: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Financial documents için özelleştirilmiş arama metodu.
        
        Args:
            query_text: Sorgu metni
            top_k: Döndürülecek en benzer belge sayısı
            financial_filters: Finansal filtreler (years, currency, table_type vs.)
            prioritize_tables: Tablo belgelerini öncelendir
            
        Returns:
            En benzer finansal belgelerin listesi
        """
        try:
            # Financial query enhancement
            enhanced_query = self._enhance_financial_query(query_text)
            
            # Initialize base filters
            base_filters = {}
            
            # Apply financial filters - ChromaDB doğru syntax kullanmalı
            if financial_filters:
                filter_conditions = []
                
                # Year filter
                if 'years' in financial_filters:
                    filter_conditions.append({
                        "financial_years": {"$in": [financial_filters['years']]}
                    })
                
                # Currency filter  
                if 'currency' in financial_filters:
                    filter_conditions.append({
                        "financial_currency": {"$eq": financial_filters['currency']}
                    })
                
                # Table type filter
                if 'table_type' in financial_filters:
                    filter_conditions.append({
                        "financial_table_type": {"$eq": financial_filters['table_type']}
                    })
                
                # Financial document filter
                if 'only_financial' in financial_filters and financial_filters['only_financial']:
                    filter_conditions.append({
                        "is_financial": {"$eq": "True"}
                    })
                
                # ChromaDB doğru filter syntax - tek koşul varsa direkt, çoklu varsa $and
                if len(filter_conditions) == 1:
                    base_filters = filter_conditions[0]
                elif len(filter_conditions) > 1:
                    base_filters = {"$and": filter_conditions}
                else:
                    base_filters = {}
            
            # İlk arama - daha geniş sonuçlar al
            initial_results = self.query_documents(
                query_text=enhanced_query,
                top_k=top_k * 4,  # Daha fazla sonuç al, sonra filtrele
                filter_criteria=base_filters if base_filters else None
            )
            
            # Results processing and scoring
            scored_results = []
            
            for result in initial_results:
                metadata = result.get('metadata', {})
                score = result.get('distance', 1.0)  # Lower is better for distance
                
                # Financial relevance scoring
                financial_score = 0.0
                
                # Table priority scoring
                if metadata.get('type') == 'table' and prioritize_tables:
                    financial_score += 0.2
                
                # Financial table type scoring
                if metadata.get('is_financial', False):
                    financial_score += 0.3
                
                # Data quality scoring
                data_quality = metadata.get('data_quality_score', 0.0)
                financial_score += data_quality * 0.2
                
                # Financial terms relevance
                found_terms = metadata.get('financial_terms_found', [])
                if found_terms:
                    # Check if query contains any of the found financial terms
                    query_lower = query_text.lower()
                    term_matches = sum(1 for term in found_terms if term.lower() in query_lower)
                    financial_score += (term_matches / len(found_terms)) * 0.3
                
                # Combine semantic score with financial relevance
                # Convert distance to similarity (1 - distance) and combine with financial score
                combined_score = (1 - score) * 0.7 + financial_score * 0.3
                
                result['financial_score'] = financial_score
                result['combined_score'] = combined_score
                scored_results.append(result)
            
            # Sort by combined score (descending)
            scored_results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Return top results
            final_results = scored_results[:top_k]
            
            logger.info(f"Financial query completed: {len(final_results)} results returned")
            return final_results
            
        except Exception as e:
            logger.error(f"Financial query error: {str(e)}")
            # Fallback to regular query
            return self.query_documents(query_text, top_k)
    
    def _enhance_financial_query(self, query_text: str) -> str:
        """
        Enhance financial queries with domain-specific terms.
        
        Args:
            query_text: Original query text
            
        Returns:
            Enhanced query text
        """
        # Financial term mappings
        financial_mappings = {
            'varlık': 'varlık asset aktif',
            'toplam varlık': 'toplam varlık total assets',
            'yükümlülük': 'yükümlülük liability borç',
            'özsermaye': 'özsermaye equity sermaye',
            'gelir': 'gelir revenue hasılat satış',
            'gider': 'gider expense maliyet',
            'kar': 'kar profit net kar',
            'zarar': 'zarar loss net zarar',
            'nakit': 'nakit cash',
            'bilanço': 'bilanço balance sheet mali durum',
            'gelir tablosu': 'gelir tablosu income statement kar zarar',
            'nakit akış': 'nakit akış cash flow'
        }
        
        enhanced_query = query_text
        
        # Add financial synonyms to query
        for original, enhanced in financial_mappings.items():
            if original in query_text.lower():
                enhanced_query += f" {enhanced}"
        
        return enhanced_query
    
    def get_financial_summary(self) -> Dict[str, Any]:
        """
        Get summary of financial documents in the vector store.
        
        Returns:
            Summary statistics of financial documents
        """
        try:
            # Get all documents to analyze
            all_results = self.collection.get()
            
            summary = {
                'total_documents': len(all_results['ids']) if all_results['ids'] else 0,
                'financial_documents': 0,
                'table_documents': 0,
                'years_covered': set(),
                'currencies_found': set(),
                'financial_table_types': {},
                'companies': set(),
                'data_quality_average': 0.0
            }
            
            quality_scores = []
            
            if all_results['metadatas']:
                for metadata in all_results['metadatas']:
                    # Count financial documents
                    if metadata.get('is_financial', False):
                        summary['financial_documents'] += 1
                    
                    # Count table documents
                    if metadata.get('type') == 'table':
                        summary['table_documents'] += 1
                    
                    # Collect years
                    years = metadata.get('financial_years', [])
                    if years:
                        summary['years_covered'].update(years)
                    
                    # Collect currencies
                    currency = metadata.get('financial_currency')
                    if currency:
                        summary['currencies_found'].add(currency)
                    
                    # Collect table types
                    table_type = metadata.get('financial_table_type', 'general')
                    if table_type in summary['financial_table_types']:
                        summary['financial_table_types'][table_type] += 1
                    else:
                        summary['financial_table_types'][table_type] = 1
                    
                    # Collect companies
                    source = metadata.get('source', '')
                    if source:
                        summary['companies'].add(source)
                    
                    # Collect quality scores
                    quality = metadata.get('data_quality_score', 0.0)
                    if quality > 0:
                        quality_scores.append(quality)
            
            # Convert sets to lists for JSON serialization
            summary['years_covered'] = sorted(list(summary['years_covered']))
            summary['currencies_found'] = list(summary['currencies_found'])
            summary['companies'] = list(summary['companies'])
            
            # Calculate average data quality
            if quality_scores:
                summary['data_quality_average'] = sum(quality_scores) / len(quality_scores)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting financial summary: {str(e)}")
            return {} 