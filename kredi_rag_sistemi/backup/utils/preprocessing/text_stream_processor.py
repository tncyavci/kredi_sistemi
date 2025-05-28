"""
Metin akış işleme modülü.
Büyük metinleri akış tabanlı işlemek için kullanılan sınıf ve yardımcı işlevler.
"""

import os
import json
import logging
from typing import List, Dict, Any, Callable, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class TextStreamProcessor:
    """
    Büyük metinleri akış tabanlı işlemek için kullanılan sınıf.
    Metin parçalarını JSON dosyasına bellek verimli şekilde yazar.
    """
    
    def __init__(
        self,
        output_path: str,
        chunk_size: int = 1000,
        overlap: int = 200,
        cleanup_funcs: Optional[List[Callable[[str], str]]] = None,
        buffer_size: int = 5000
    ):
        """
        TextStreamProcessor sınıfının başlatıcısı.
        
        Args:
            output_path: Çıktı JSON dosyasının yolu
            chunk_size: Metin parçalama uzunluğu (kelime sayısı)
            overlap: Parçalar arası örtüşme (kelime sayısı)
            cleanup_funcs: Metin temizleme fonksiyonları listesi
            buffer_size: Tampon boyutu (kelime sayısı)
        """
        self.output_path = Path(output_path)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.cleanup_funcs = cleanup_funcs or []
        self.buffer_size = buffer_size
        
        # Akış işleme için değişkenler
        self.buffer = []  # Kelime listesi
        self.chunks = []  # Oluşturulan parçalar
        self.chunk_count = 0
        self.total_words = 0
        
        # Çıktı dosyası başlangıcını yaz
        self._init_output_file()
    
    def _init_output_file(self):
        """
        Çıktı JSON dosyasını başlatır.
        Dosyanın JSON başlangıç yapısını yazar.
        """
        # Dizini oluştur
        self.output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # JSON başlangıç yapısını yaz
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write('{\n')
            f.write('  "chunks": [\n')
    
    def _append_chunk_to_file(self, chunk: Dict[str, Any], is_last: bool = False):
        """
        Metin parçasını çıktı dosyasına ekler.
        
        Args:
            chunk: Eklenecek metin parçası
            is_last: Son parça mı?
        """
        with open(self.output_path, 'a', encoding='utf-8') as f:
            json_chunk = json.dumps(chunk, ensure_ascii=False, indent=2)
            
            # Virgül ekle (son parça değilse)
            if not is_last:
                f.write(f"{json_chunk},\n")
            else:
                f.write(f"{json_chunk}\n")
    
    def _finalize_output_file(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Çıktı dosyasını tamamlar.
        
        Args:
            metadata: JSON dosyasına eklenecek ek meta veriler
        """
        with open(self.output_path, 'a', encoding='utf-8') as f:
            f.write('  ],\n')
            
            # Meta verileri ekle
            if metadata:
                meta_json = json.dumps(metadata, ensure_ascii=False, indent=2)
                # İlk ve son süslü parantezleri kaldır
                meta_json = meta_json[1:-1].strip()
                f.write(f"  {meta_json}\n")
            
            f.write('}\n')
    
    def process_text_chunk(self, text: str) -> List[Dict[str, Any]]:
        """
        Bir metin parçasını işler ve tampondaki kelimelere ekler.
        Yeterli kelime biriktiğinde, yeni parçalar oluşturur.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            Oluşturulan metin parçaları
        """
        # Metni temizle
        for cleanup_func in self.cleanup_funcs:
            text = cleanup_func(text)
        
        # Kelimelere böl ve tampona ekle
        words = text.split()
        self.buffer.extend(words)
        self.total_words += len(words)
        
        # Yeterli kelime varsa parçalar oluştur
        new_chunks = []
        while len(self.buffer) >= self.chunk_size:
            # Parça için kelimeleri al
            chunk_words = self.buffer[:self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Parçayı oluştur
            chunk = {
                "text": chunk_text,
                "length": len(chunk_words),
                "byte_size": len(chunk_text.encode('utf-8')),
                "chunk_id": self.chunk_count
            }
            
            # Parçayı ekle
            self.chunks.append(chunk)
            new_chunks.append(chunk)
            self._append_chunk_to_file(chunk)
            self.chunk_count += 1
            
            # Tamponu güncelle (örtüşmeyi koru)
            self.buffer = self.buffer[self.chunk_size - self.overlap:]
        
        return new_chunks
    
    def finalize(self, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        İşlemeyi tamamlar ve son parçayı ekler.
        
        Args:
            metadata: JSON dosyasına eklenecek ek meta veriler
            
        Returns:
            Toplam parça sayısı
        """
        # Tamponda kalan kelimeleri son parça olarak ekle
        if self.buffer:
            chunk_text = " ".join(self.buffer)
            
            # Son parçayı oluştur
            chunk = {
                "text": chunk_text,
                "length": len(self.buffer),
                "byte_size": len(chunk_text.encode('utf-8')),
                "chunk_id": self.chunk_count
            }
            
            # Son parçayı ekle
            self.chunks.append(chunk)
            self._append_chunk_to_file(chunk, is_last=True)
            self.chunk_count += 1
        else:
            # Parça yoksa, son virgülü kaldır
            if self.chunks:
                # Dosya açık değilse, son ekleme işlemini düzelt
                with open(self.output_path, 'rb+') as f:
                    f.seek(-2, os.SEEK_END)  # Son virgülü ve yeni satırı atla
                    f.truncate()
                
                with open(self.output_path, 'a', encoding='utf-8') as f:
                    f.write('\n')
        
        # Meta verileri ekle ve dosyayı tamamla
        if metadata is None:
            metadata = {}
        
        # Temel istatistikler
        metadata.update({
            "total_chunks": self.chunk_count,
            "total_words": self.total_words,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap
        })
        
        # Dosyayı tamamla
        self._finalize_output_file(metadata)
        
        logger.info(f"Akış işleme tamamlandı: {self.chunk_count} parça, {self.total_words} kelime")
        return self.chunk_count 