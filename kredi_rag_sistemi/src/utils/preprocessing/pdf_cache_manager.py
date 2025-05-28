"""
PDF işleme sonuçları için önbellekleme yöneticisi.
İşlenmiş PDF'leri önbelleğe alarak tekrar işleme gereksinimini azaltır.
"""

import os
import json
import hashlib
import shutil
import logging
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

class PDFCacheManager:
    """
    PDF işleme sonuçları için önbellekleme yöneticisi.
    İşlenmiş PDF'lerin önbelleğe alınması ve yönetilmesini sağlar.
    """
    
    def __init__(
        self,
        cache_dir: str = "./data/pdf_cache",
        max_cache_size_mb: int = 5000,  # 5GB
        max_cache_age_days: int = 30,
        use_compression: bool = True
    ):
        """
        PDFCacheManager'ı başlatır
        
        Args:
            cache_dir: Önbellek dizini
            max_cache_size_mb: Maksimum önbellek boyutu (MB)
            max_cache_age_days: Önbellekteki öğelerin maksimum yaşı (gün)
            use_compression: Sıkıştırma kullanılsın mı
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        self.metadata_path = self.cache_dir / "metadata.json"
        self.max_cache_size_mb = max_cache_size_mb
        self.max_cache_age_days = max_cache_age_days
        self.use_compression = use_compression
        
        # Metadata yükleme veya oluşturma
        self.metadata = self._load_metadata()
        
        # Önbelleği temizle (eski veya fazla alan kaplayan)
        self._clean_cache()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Önbellek metadata bilgisini yükler"""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Önbellek metadata dosyası yüklenemedi: {str(e)}")
                
        # Varsayılan metadata
        return {
            "items": {},
            "total_size_bytes": 0,
            "last_cleanup": time.time()
        }
    
    def _save_metadata(self):
        """Önbellek metadata bilgisini kaydeder"""
        try:
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Önbellek metadata kaydedilemedi: {str(e)}")
    
    def _get_cache_key(self, pdf_path: str, params: Dict[str, Any]) -> str:
        """
        PDF dosyası ve işleme parametreleri için önbellek anahtarı oluşturur
        
        Args:
            pdf_path: PDF dosyasının yolu
            params: İşleme parametreleri
            
        Returns:
            Önbellek anahtarı
        """
        # Dosyanın son değiştirilme zamanını al
        try:
            mtime = os.path.getmtime(pdf_path)
        except FileNotFoundError:
            mtime = 0
        
        # PDF dosya yolu ve mtime ile parametre değerlerini birleştir
        content = f"{pdf_path}|{mtime}|{json.dumps(params, sort_keys=True)}"
        
        # SHA-256 hash oluştur
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Önbellek anahtarı için dosya yolunu döndürür"""
        return self.cache_dir / f"{cache_key}.{'pkl.gz' if self.use_compression else 'pkl'}"
    
    def _clean_cache(self):
        """
        Eski ve fazla alan kaplayan önbellek öğelerini temizler
        """
        # Son temizlikten beri yeterli zaman geçtiyse temizle
        current_time = time.time()
        if (current_time - self.metadata.get("last_cleanup", 0)) < 3600:  # 1 saat
            return
            
        logger.info("Önbellek temizliği başlatılıyor...")
        
        # Eski öğeleri temizle
        items_to_remove = []
        max_age_seconds = self.max_cache_age_days * 24 * 3600
        
        for key, item in self.metadata["items"].items():
            age = current_time - item.get("timestamp", 0)
            if age > max_age_seconds:
                items_to_remove.append(key)
        
        # Öğeleri kaldır
        for key in items_to_remove:
            self._remove_cache_item(key)
            
        # Boyut aşılmışsa en eski öğelerden başlayarak temizle
        total_size_mb = self.metadata["total_size_bytes"] / (1024 * 1024)
        if total_size_mb > self.max_cache_size_mb:
            # Öğeleri erişim zamanına göre sırala
            sorted_items = sorted(
                self.metadata["items"].items(),
                key=lambda x: x[1].get("last_access", 0)
            )
            
            # Boyut sınırı altına düşene kadar temizle
            for key, _ in sorted_items:
                self._remove_cache_item(key)
                
                # Yeni boyutu kontrol et
                total_size_mb = self.metadata["total_size_bytes"] / (1024 * 1024)
                if total_size_mb <= self.max_cache_size_mb * 0.9:  # %10 marj bırak
                    break
        
        # Son temizleme zamanını güncelle
        self.metadata["last_cleanup"] = current_time
        self._save_metadata()
        
        logger.info(f"Önbellek temizliği tamamlandı. Yeni boyut: {total_size_mb:.2f} MB")
    
    def _remove_cache_item(self, cache_key: str):
        """Önbellekten bir öğeyi kaldırır"""
        if cache_key not in self.metadata["items"]:
            return
            
        # Dosyayı sil
        cache_file = self._get_cache_file_path(cache_key)
        try:
            if cache_file.exists():
                item_size = cache_file.stat().st_size
                cache_file.unlink()
                
                # Metadata güncelle
                self.metadata["total_size_bytes"] -= item_size
                del self.metadata["items"][cache_key]
        except Exception as e:
            logger.error(f"Önbellek öğesi silinemedi: {str(e)}")
    
    def get(self, pdf_path: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Önbellekten bir PDF işleme sonucunu alır
        
        Args:
            pdf_path: PDF dosyasının yolu
            params: İşleme parametreleri
            
        Returns:
            Önbellekten bulunan işleme sonucu veya None
        """
        cache_key = self._get_cache_key(pdf_path, params)
        
        if cache_key not in self.metadata["items"]:
            return None
            
        cache_file = self._get_cache_file_path(cache_key)
        if not cache_file.exists():
            # Dosya yoksa metadata'dan da kaldır
            if cache_key in self.metadata["items"]:
                del self.metadata["items"][cache_key]
                self._save_metadata()
            return None
        
        try:
            # Veriyi yükle
            if self.use_compression:
                import gzip
                with gzip.open(cache_file, 'rb') as f:
                    result = pickle.load(f)
            else:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
            
            # Son erişim zamanını güncelle
            self.metadata["items"][cache_key]["last_access"] = time.time()
            self._save_metadata()
            
            logger.info(f"PDF dosyası önbellekten yüklendi: {pdf_path}")
            return result
            
        except Exception as e:
            logger.warning(f"Önbellekten PDF yüklenirken hata: {str(e)}")
            return None
    
    def set(self, pdf_path: str, params: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """
        PDF işleme sonucunu önbelleğe kaydeder
        
        Args:
            pdf_path: PDF dosyasının yolu
            params: İşleme parametreleri
            result: Kaydedilecek sonuç
            
        Returns:
            İşlemin başarılı olup olmadığı
        """
        if not result:
            return False
            
        cache_key = self._get_cache_key(pdf_path, params)
        cache_file = self._get_cache_file_path(cache_key)
        
        try:
            # Veriyi sıkıştırarak veya sıkıştırmadan kaydet
            if self.use_compression:
                import gzip
                with gzip.open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            else:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            
            # Metadata güncelle
            file_size = cache_file.stat().st_size
            
            # Eğer bu anahtar zaten varsa eski boyutu çıkar
            if cache_key in self.metadata["items"]:
                self.metadata["total_size_bytes"] -= self.metadata["items"][cache_key].get("size", 0)
            
            self.metadata["items"][cache_key] = {
                "size": file_size,
                "timestamp": time.time(),
                "last_access": time.time(),
                "pdf_path": pdf_path
            }
            
            self.metadata["total_size_bytes"] += file_size
            self._save_metadata()
            
            logger.info(f"PDF dosyası önbelleğe kaydedildi: {pdf_path}")
            return True
            
        except Exception as e:
            logger.error(f"PDF önbelleğe kaydedilirken hata: {str(e)}")
            return False
            
    def invalidate(self, pdf_path: Optional[str] = None):
        """
        Belirli bir PDF dosyası veya tüm önbelleği geçersiz kılar
        
        Args:
            pdf_path: Geçersiz kılınacak PDF dosyasının yolu (None ise tümü)
        """
        if pdf_path:
            # Belirli bir PDF dosyasını geçersiz kıl
            items_to_remove = []
            for key, item in self.metadata["items"].items():
                if item.get("pdf_path") == pdf_path:
                    items_to_remove.append(key)
            
            for key in items_to_remove:
                self._remove_cache_item(key)
                
            logger.info(f"PDF dosyası önbellekten kaldırıldı: {pdf_path}")
        else:
            # Tüm önbelleği temizle
            try:
                for item in self.cache_dir.glob("*.pkl*"):
                    item.unlink()
                
                # Metadata'yı sıfırla
                self.metadata = {
                    "items": {},
                    "total_size_bytes": 0,
                    "last_cleanup": time.time()
                }
                self._save_metadata()
                
                logger.info("Tüm önbellek temizlendi")
            except Exception as e:
                logger.error(f"Önbellek temizlenirken hata: {str(e)}")
                
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Önbellek istatistiklerini döndürür
        
        Returns:
            Önbellek istatistikleri
        """
        total_size_mb = self.metadata["total_size_bytes"] / (1024 * 1024)
        
        return {
            "item_count": len(self.metadata["items"]),
            "total_size_mb": round(total_size_mb, 2),
            "max_size_mb": self.max_cache_size_mb,
            "usage_percent": round((total_size_mb / self.max_cache_size_mb) * 100, 2) if self.max_cache_size_mb > 0 else 0,
            "last_cleanup": self.metadata.get("last_cleanup", 0)
        } 