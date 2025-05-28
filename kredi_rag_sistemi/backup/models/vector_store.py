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
            # Eski: self.client = chromadb.Client(settings=chroma_settings)
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False  # Telemetri devre dışı (güvenlik için)
                )
            )
            
            # ÇÖZÜM: ChromaDB'ye kendi embedding fonksiyonumuzu veriyoruz
            # Doğrudan fonksiyon vermek yerine kendi sınıfımızı oluşturuyoruz
            # Bu, meta tensor sorununu çözecek
            
            class SafeEmbeddingFunction(chromadb.EmbeddingFunction):
                """
                ChromaDB için güvenli embedding fonksiyonu.
                Doğrudan SentenceTransformers'ı CPU'da kullanıyor ve meta tensor hatasını çözüyor.
                """
                
                def __init__(self, model_name):
                    from sentence_transformers import SentenceTransformer
                    import os
                    import torch
                    
                    self.model_name = model_name
                    
                    try:
                        # Model önbelleği için dizin oluştur
                        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                               "models", "embeddings", ".cache")
                        os.makedirs(cache_dir, exist_ok=True)
                        
                        # Tokenizer ve model ayarları
                        os.environ["TOKENIZERS_PARALLELISM"] = "false"
                        os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir
                        
                        # MPS ve CUDA'yı devre dışı bırak (Apple Silicon için)
                        os.environ["CUDA_VISIBLE_DEVICES"] = ""
                        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                        
                        # Meta tensor hatasını çözmek için to_empty() ve CPU'ya taşıma
                        logger.info(f"SentenceTransformer modeli yükleniyor: {model_name}")
                        
                        try:
                            # MPS ve CUDA'yı devre dışı bırak
                            os.environ["CUDA_VISIBLE_DEVICES"] = ""
                            # PYTORCH_ENABLE_MPS_FALLBACK artık gerekmeyebilir,
                            # çünkü doğrudan CPU belirtiyoruz.
                            # os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" 

                            logger.info(f"SentenceTransformer modeli CPU'da yükleniyor: {model_name}")
                            # Doğrudan CPU'da başlat
                            self.model = SentenceTransformer(model_name, device="cpu", cache_folder=cache_dir)
                            logger.info(f"SentenceTransformer modeli başarıyla CPU'da yüklendi: {model_name}")
                            
                        except Exception as e1:
                            # Direkt olarak CPU'da başlatmayı dene
                            logger.warning(f"to_empty() yöntemi başarısız, direkt CPU'da deneniyor: {str(e1)}")
                            self.model = SentenceTransformer(model_name, device="cpu", cache_folder=cache_dir)
                            
                        logger.info(f"SentenceTransformer modeli başarıyla yüklendi: {model_name}")
                        
                    except Exception as e:
                        logger.error(f"Ana model yüklenemedi: {str(e)}")
                        # Alternatif model dene
                        try:
                            backup_model = "sentence-transformers/all-MiniLM-L6-v2"  # Daha küçük ve daha kararlı model
                            logger.warning(f"Alternatif model deneniyor: {backup_model}")
                            
                            # Doğrudan CPU'da alternatif modeli dene
                            self.model = SentenceTransformer(backup_model, device="cpu", cache_folder=cache_dir)
                            self.model_name = backup_model
                            logger.info(f"Alternatif model başarıyla yüklendi: {backup_model}")
                        except Exception as e2:
                            logger.error(f"Alternatif model de yüklenemedi: {str(e2)}")
                            raise
                
                def __call__(self, texts):
                    """Metinleri vektörlere dönüştürür, bellek yönetimi ile"""
                    import torch
                    import gc
                    
                    try:
                        # Daha küçük batch boyutu kullan
                        batch_size = 16  # Daha küçük batch boyutu, bellek için daha güvenli
                        all_embeddings = []
                        
                        # Metinleri batch'lere bölerek işle
                        for i in range(0, len(texts), batch_size):
                            batch = texts[i:i + batch_size]
                            
                            # CUDA ve MPS önbelleğini temizle
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                            # Kodlama yapmadan önce GC çalıştır
                            gc.collect()
                            
                            # CPU'da embeddinglari hesapla
                            with torch.no_grad():  # Bellek kullanımını azaltmak için
                                embeddings = self.model.encode(batch, convert_to_numpy=True)
                            
                            all_embeddings.extend(embeddings.tolist())
                            
                            # Her batch'ten sonra belleği temizle
                            del embeddings
                            gc.collect()
                            
                        return all_embeddings
                    except Exception as e:
                        logger.error(f"Embedding oluşturma hatası: {str(e)}")
                        # Tek tek deneyerek hatalı metni bul ve atla
                        all_embeddings = []
                        for text in texts:
                            try:
                                # Belleği temizle
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    
                                with torch.no_grad():
                                    embedding = self.model.encode(text, convert_to_numpy=True)
                                all_embeddings.append(embedding.tolist())
                            except Exception as e:
                                logger.error(f"Metin kodlanamadı: {text[:50]}... Hata: {str(e)}")
                                # Varsayılan 0 vektörü ekle (aynı boyutta)
                                dims = 384  # Modele göre değişebilir
                                if self.model_name == "sentence-transformers/all-MiniLM-L6-v2":
                                    dims = 384
                                elif self.model_name == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2":
                                    dims = 384
                                all_embeddings.append([0.0] * dims)
                        
                        return all_embeddings
            
            # Özel embedding fonksiyonu oluştur - sadece all-MiniLM-L6-v2 modeli kullan
            self.embedding_function = SafeEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")
            logger.info(f"Özel güvenli embedding fonksiyonu oluşturuldu: all-MiniLM-L6-v2")
            
            # Koleksiyonu oluştur veya yükle
            self._create_or_load_collection()
            
            logger.info(f"ChromaDB koleksiyonu başarıyla kuruldu: {self.collection_name}")
        except Exception as e:
            logger.error(f"ChromaDB kurulum hatası: {str(e)}")
            raise
    
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
        """Belgeleri ChromaDB'ye ekler"""
        # ChromaDB'nin işleyebileceği maksimum batch boyutu (güvenli değer)
        max_batch_size = 500
        
        # Belgeleri daha küçük batch'lere böl
        for i in range(0, len(documents), max_batch_size):
            batch = documents[i:i + max_batch_size]
            
            ids = [doc["id"] for doc in batch]
            texts = [doc["text"] for doc in batch]
            embeddings = [doc.get("embedding") for doc in batch if "embedding" in doc]
            # Metadata'ları temizle
            metadatas = [self._sanitize_metadata(doc.get("metadata", {})) for doc in batch]
            
            # Embeddingler sağlanmışsa kullan, yoksa Chroma kendi hesaplayacak
            try:
                if embeddings and all(emb is not None for emb in embeddings):
                    self.collection.add(
                        ids=ids,
                        documents=texts,
                        embeddings=embeddings,
                        metadatas=metadatas
                    )
                else:
                    self.collection.add(
                        ids=ids,
                        documents=texts,
                        metadatas=metadatas
                    )
                
                logger.info(f"ChromaDB'ye {len(batch)} belge eklendi. (Toplam batch: {i//max_batch_size + 1})")
            except Exception as e:
                logger.error(f"ChromaDB belge ekleme hatası (batch {i//max_batch_size + 1}): {str(e)}")
                # Tekil ekleme stratejisine geç
                logger.info("Tekil belge ekleme stratejisine geçiliyor...")
                for j, doc in enumerate(batch):
                    try:
                        doc_id = doc["id"]
                        text = doc["text"]
                        metadata = self._sanitize_metadata(doc.get("metadata", {}))
                        embedding = doc.get("embedding")
                        
                        if embedding is not None:
                            self.collection.add(
                                ids=[doc_id],
                                documents=[text],
                                embeddings=[embedding],
                                metadatas=[metadata]
                            )
                        else:
                            self.collection.add(
                                ids=[doc_id],
                                documents=[text],
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
        """Koleksiyon oluştur veya varsa yükle"""
        try:
            # Önce koleksiyonun var olup olmadığını kontrol et
            collections = self.client.list_collections()
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if collection_exists and not self.force_recreate:
                # Varolan koleksiyonu yükle
                logger.info(f"Varolan ChromaDB koleksiyonu kullanılıyor: {self.collection_name}")
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
            else:
                if collection_exists and self.force_recreate:
                    # Koleksiyonu sil (zorla yeniden oluşturulacaksa)
                    logger.warning(f"Mevcut koleksiyon bulundu ve silinecek: {self.collection_name}")
                    self.client.delete_collection(name=self.collection_name)
                    logger.info(f"Eski koleksiyon silindi: {self.collection_name}")
                
                # Yeni koleksiyon oluştur
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"description": "Kredi RAG sistemi için vektör veritabanı"}
                )
                logger.info(f"Yeni ChromaDB koleksiyonu oluşturuldu: {self.collection_name}")
            
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
        Vektör veritabanında sorgu yapar
        
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
            expanded_top_k = top_k * 3
            
            # Sorguyu yap
            query_results = self.collection.query(
                query_texts=[query_text],
                n_results=expanded_top_k,
                where=filter_criteria
            )
            
            # Sonuçları formatlı hale getir
            results = []
            if query_results and len(query_results['ids'][0]) > 0:
                for i in range(len(query_results['ids'][0])):
                    # Sonuç kalitesini kontrol et
                    distance = query_results['distances'][0][i] if 'distances' in query_results else 0
                    
                    # Distance threshold'u düşürdük - tablo belgeleri için daha esnek
                    # Sadece çok düşük benzerliğe sahip belgeleri hariç tut
                    if distance > 2.5 and top_k > 1:  # Threshold 1.8'den 2.5'e çıkarıldı (tablo belgeleri için)
                        continue
                        
                    result = {
                        "id": query_results['ids'][0][i],
                        "text": query_results['documents'][0][i],
                        "metadata": query_results['metadatas'][0][i],
                        "distance": distance
                    }
                    results.append(result)
            
            # En iyi top_k sonucu döndür
            results = results[:top_k]
            
            logger.info(f"Sorgu başarıyla çalıştırıldı, {len(results)} sonuç döndürüldü")
            return results
            
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