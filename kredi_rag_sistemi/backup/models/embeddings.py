from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Union
import os
import pickle
import gc
import logging
import faiss  # FAISS ekle
import torch  # PyTorch ekleniyor

logger = logging.getLogger(__name__)

class DocumentEmbedder:
    """
    Dökümanları vektörlere dönüştürmek ve vektör veritabanı oluşturmak için kullanılan sınıf.
    Kredi belgeleri için metin gömme (text embedding) işlemlerini yönetir.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",  # Daha kompakt olan model
        cache_dir: str = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        memory_cleanup_threshold: int = 100,
        use_faiss: bool = True  # FAISS kullanımı için bayrak
    ):
        """
        DocumentEmbedder sınıfının başlatıcısı.
        
        Args:
            model_name: Kullanılacak metin gömme modelinin adı
            cache_dir: Model önbelleği için dizin (isteğe bağlı)
            chunk_size: Metin parçalanacak maksimum uzunluk
            chunk_overlap: Metin parçalanırken örtüşen kelime sayısı
            memory_cleanup_threshold: Bellek temizliği için eşik değeri
            use_faiss: FAISS vektör indeksi kullanılacak mı
        """
        try:
            # Cache dizinini proje içinde tutmak için ayarla
            if cache_dir is None:
                script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.cache_dir = os.path.join(script_dir, "models", "embeddings", ".cache")
                os.makedirs(self.cache_dir, exist_ok=True)
                logger.info(f"Model cache dizini: {self.cache_dir}")
            else:
                self.cache_dir = cache_dir
                os.makedirs(self.cache_dir, exist_ok=True)
            
            # Tokenizer ve model ayarları
            os.environ["TOKENIZERS_PARALLELISM"] = "false"  
            os.environ["SENTENCE_TRANSFORMERS_HOME"] = self.cache_dir
            
            # META TENSOR HATASI İÇİN SADECE CPU'DA ÇALIŞ - MPS/CUDA KULLANMA
            logger.info(f"SADECE CPU kullanılacak (meta tensor hatasını önlemek için)")
            
            # SentenceTransformer model önbelleğini ayarla
            # Modeli açıkça belirtilen önbellek dizininden yükle
            # Cihaz atama sorununu çözmek için her zaman CPU'da çalış
            try:
                # CPU'da başlat - device parametresini asla değiştirme
                self.model = SentenceTransformer(
                    model_name, 
                    cache_folder=self.cache_dir,
                    device="cpu"  # Sadece CPU kullan
                )
                
                logger.info(f"Model başarıyla yüklendi: {model_name}")
            except Exception as e:
                logger.error(f"Model yüklenirken hata oluştu: {str(e)}")
                # Alternatif model deneme
                try:
                    backup_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                    logger.warning(f"Alternatif model denenecek: {backup_model}")
                    self.model = SentenceTransformer(
                        backup_model, 
                        cache_folder=self.cache_dir,
                        device="cpu"  # Sadece CPU kullan
                    )
                    logger.info(f"Alternatif model başarıyla yüklendi: {backup_model}")
                except Exception as e2:
                    logger.error(f"Alternatif model yüklenirken de hata oluştu: {str(e2)}")
                    raise
            
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.memory_cleanup_threshold = memory_cleanup_threshold
            self.use_faiss = use_faiss
            
            self.vector_db = {}  # Vektor veritabanı
            
            # FAISS indeksi
            self.faiss_index = None
            self.embedding_dim = None
            
        except Exception as e:
            logger.error(f"Model yüklenirken hata oluştu: {str(e)}")
            raise
            
    def _chunk_text(self, text: str) -> List[str]:
        """Metni belirli uzunlukta parçalara böler"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= self.chunk_size:
                current_chunk.append(word)
                current_length += len(word) + 1
            else:
                chunks.append(" ".join(current_chunk))
                # Örtüşme için son kelimeleri yeni chunk'a ekle
                overlap_words = current_chunk[-self.chunk_overlap:]
                current_chunk = overlap_words + [word]
                current_length = sum(len(w) + 1 for w in current_chunk)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Verilen metni vektöre dönüştürür.
        
        Args:
            text: Gömme işlemi yapılacak metin
            
        Returns:
            Metin vektörü
        """
        try:
            return self.model.encode(text)
        except Exception as e:
            logger.error(f"Metin kodlama hatası: {str(e)}")
            # Yeniden deneme
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            return self.model.encode(text)
    
    def _cleanup_memory(self, force: bool = False):
        """Belleği temizler"""
        gc.collect()
        # GPU belleğini de temizle
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if force:
            # Daha agresif temizleme
            for key in list(self.vector_db.keys()):
                if isinstance(self.vector_db[key]['embedding'], np.ndarray):
                    self.vector_db[key]['embedding'] = None
            gc.collect()
    
    def _initialize_faiss(self, embedding_dim: int):
        """FAISS indeksini başlatır"""
        if self.use_faiss:
            self.embedding_dim = embedding_dim
            self.faiss_index = faiss.IndexFlatL2(embedding_dim)
            
    def _update_faiss_index(self, embedding: np.ndarray, doc_id: str):
        """FAISS indeksini günceller"""
        if self.use_faiss and self.faiss_index is not None:
            # Embedding'i normalize et
            faiss.normalize_L2(embedding.reshape(1, -1))
            self.faiss_index.add(embedding.reshape(1, -1))
            
    def embed_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Belge listesini embeddingler ve veritabanına ekler.
        
        Args:
            documents: Belge listesi. Her belge {"id": "...", "text": "...", "metadata": {...}} formatında olmalı
            
        Returns:
            Güncellenmiş vektör veritabanı
        """
        processed_count = 0
        embeddings_list = []
        doc_ids = []
        
        # Batch işleme boyutu
        batch_size = 10  # Daha büyük boyut için daha hızlı işleme, ama daha çok bellek
        
        # Belgeleri batch'lere böl
        document_batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        
        for batch in document_batches:
            batch_processed = 0
            
            for doc in batch:
                try:
                    doc_id = doc.get("id")
                    text = doc.get("text")
                    metadata = doc.get("metadata", {})
                    
                    if not doc_id or not text:
                        continue
                    
                    # Metni parçalara böl
                    chunks = self._chunk_text(text)
                    
                    # Her parçayı ayrı bir belge olarak ekle
                    for i, chunk in enumerate(chunks):
                        chunk_id = f"{doc_id}_chunk_{i}"
                        embedding = self.embed_text(chunk)
                        
                        # İlk embedding için FAISS'i başlat
                        if self.use_faiss and self.faiss_index is None:
                            self._initialize_faiss(embedding.shape[0])
                        
                        # FAISS indeksini güncelle
                        if self.use_faiss:
                            self._update_faiss_index(embedding, chunk_id)
                            embeddings_list.append(embedding)
                            doc_ids.append(chunk_id)
                        
                        self.vector_db[chunk_id] = {
                            "text": chunk,
                            "embedding": embedding,
                            "metadata": {**metadata, "chunk_index": i, "original_id": doc_id}
                        }
                    
                    batch_processed += 1
                    processed_count += 1
                    
                    # İşlem durumunu raporla
                    if processed_count % 100 == 0:
                        logger.info(f"{processed_count}/{len(documents)} belge işlendi...")
                        
                except Exception as e:
                    logger.error(f"Belge işlenirken hata oluştu (ID: {doc.get('id', 'unknown')}): {str(e)}")
                    continue
            
            # Her batch sonrası belleği temizle
            if batch_processed > 0:
                self._cleanup_memory(force=False)
        
        # Son belleği temizle
        self._cleanup_memory(force=True)
        logger.info(f"Toplam {processed_count} belge işlendi ve {len(self.vector_db)} vektör oluşturuldu.")
        return self.vector_db
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Belirli bir sorgu için en benzer belgeleri bulur.
        
        Args:
            query: Arama sorgusu
            top_k: Döndürülecek en benzer belge sayısı
            
        Returns:
            En benzer belgelerin listesi, benzerlik puanlarıyla birlikte
        """
        try:
            query_embedding = self.embed_text(query)
            
            if self.use_faiss and self.faiss_index is not None:
                # FAISS ile arama
                query_embedding = query_embedding.reshape(1, -1)
                faiss.normalize_L2(query_embedding)
                distances, indices = self.faiss_index.search(query_embedding, top_k)
                
                # Sonuçları dönüştür
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx != -1:  # Geçerli bir indeks
                        doc_id = list(self.vector_db.keys())[idx]
                        doc_data = self.vector_db[doc_id]
                        results.append({
                            "id": doc_id,
                            "score": float(1 - distances[0][i]),  # L2 mesafesini benzerliğe çevir
                            "text": doc_data["text"],
                            "metadata": doc_data["metadata"]
                        })
                
                return results
                
            else:
                # Vektörel işlemler için numpy kullan
                similarities = []
                for doc_id, doc_data in self.vector_db.items():
                    doc_embedding = doc_data["embedding"]
                    if doc_embedding is not None:
                        similarity = np.dot(query_embedding, doc_embedding)
                        similarities.append({
                            "id": doc_id,
                            "score": float(similarity),
                            "text": doc_data["text"],
                            "metadata": doc_data["metadata"]
                        })
                
                sorted_results = sorted(similarities, key=lambda x: x["score"], reverse=True)
                
                # Aynı belgeden gelen parçaları birleştir
                unique_results = []
                seen_docs = set()
                
                for result in sorted_results:
                    original_id = result["metadata"].get("original_id", result["id"])
                    if original_id not in seen_docs:
                        seen_docs.add(original_id)
                        unique_results.append(result)
                        if len(unique_results) >= top_k:
                            break
                
                return unique_results
            
        except Exception as e:
            logger.error(f"Benzerlik araması sırasında hata oluştu: {str(e)}")
            return []
        finally:
            # Belleği temizle
            self._cleanup_memory()
    
    def save_vector_db(self, filepath: str) -> None:
        """
        Vektör veritabanını diske kaydeder.
        
        Args:
            filepath: Veritabanı dosyasının yolu
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.vector_db, f)
            
    def load_vector_db(self, filepath: str) -> Dict[str, Any]:
        """
        Vektör veritabanını diskten yükler.
        
        Args:
            filepath: Veritabanı dosyasının yolu
            
        Returns:
            Yüklenmiş vektör veritabanı
        """
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.vector_db = pickle.load(f)
        else:
            self.vector_db = {}
            
        return self.vector_db 