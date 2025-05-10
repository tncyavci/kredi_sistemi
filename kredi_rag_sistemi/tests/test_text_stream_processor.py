import os
import json
import pytest
import tempfile
from pathlib import Path

from utils.preprocessing.text_stream_processor import TextStreamProcessor

@pytest.fixture
def temp_output_path():
    """Geçici çıktı dosya yolu"""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name
    
    yield tmp_path
    
    # Test sonunda geçici dosyayı temizle
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

@pytest.fixture
def sample_text():
    """Test için örnek metin"""
    return "Bu bir test metnidir. " * 100

@pytest.fixture
def sample_text_chunks():
    """Test için örnek metin parçaları"""
    return [
        "Bu bir test metnidir. " * 10,
        "Bu başka bir test metnidir. " * 10,
        "Üçüncü test metni burada. " * 10
    ]

@pytest.fixture
def test_cleanup_funcs():
    """Test için metin temizleme fonksiyonları"""
    def remove_extra_spaces(text):
        return " ".join(text.split())
    
    def to_lowercase(text):
        return text.lower()
    
    return [remove_extra_spaces, to_lowercase]

class TestTextStreamProcessor:
    """TextStreamProcessor sınıfı için testler"""
    
    def test_initialization(self, temp_output_path):
        """Başlatma ve dosya oluşturma testi"""
        processor = TextStreamProcessor(
            output_path=temp_output_path,
            chunk_size=50,
            overlap=10
        )
        
        # Çıktı dosyası oluşturuldu mu?
        assert os.path.exists(temp_output_path)
        
        # Dosya içeriği doğru mu?
        with open(temp_output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert '{\n' in content
            assert '  "chunks": [\n' in content
    
    def test_process_text_chunk(self, temp_output_path, sample_text):
        """Metin parçası işleme testi"""
        processor = TextStreamProcessor(
            output_path=temp_output_path,
            chunk_size=20,
            overlap=5
        )
        
        # Metni işle
        chunks = processor.process_text_chunk(sample_text)
        
        # Parçalar oluşturuldu mu?
        assert len(chunks) > 0
        
        # Her parça doğru formatta mı?
        for chunk in chunks:
            assert "text" in chunk
            assert "length" in chunk
            assert "byte_size" in chunk
            assert "chunk_id" in chunk
            assert chunk["length"] == 20  # Chunk boyutu
    
    def test_chunk_overlap(self, temp_output_path, sample_text):
        """Metin parçaları arasındaki örtüşme testi"""
        processor = TextStreamProcessor(
            output_path=temp_output_path,
            chunk_size=20,
            overlap=5
        )
        
        # Metni işle
        chunks = processor.process_text_chunk(sample_text)
        
        # En az iki parça var mı?
        if len(chunks) >= 2:
            # İlk parçanın son kelimeleri ile ikinci parçanın ilk kelimeleri örtüşmeli
            first_chunk_words = chunks[0]["text"].split()
            second_chunk_words = chunks[1]["text"].split()
            
            # Son 5 kelime ile ilk 5 kelime örtüşmeli
            overlap_words = first_chunk_words[-5:]
            
            # Örtüşmeyi kontrol et
            for word in overlap_words:
                assert word in second_chunk_words[:10]
    
    def test_cleanup_functions(self, temp_output_path, sample_text, test_cleanup_funcs):
        """Temizleme fonksiyonları testi"""
        processor = TextStreamProcessor(
            output_path=temp_output_path,
            chunk_size=20,
            overlap=5,
            cleanup_funcs=test_cleanup_funcs
        )
        
        # Fazladan boşluk içeren metin
        test_text = "  Bu   bir  TEST   metnidir.  " * 10
        
        # Metni işle
        chunks = processor.process_text_chunk(test_text)
        
        # Temizleme fonksiyonları uygulandı mı?
        if chunks:
            # Küçük harfe dönüştürüldü mü?
            assert "TEST" not in chunks[0]["text"]
            assert "test" in chunks[0]["text"]
            
            # Fazla boşluklar temizlendi mi?
            assert "  " not in chunks[0]["text"]
    
    def test_multiple_text_chunks(self, temp_output_path, sample_text_chunks):
        """Birden fazla metin parçası işleme testi"""
        processor = TextStreamProcessor(
            output_path=temp_output_path,
            chunk_size=30,
            overlap=5
        )
        
        total_chunks = 0
        
        # Birden fazla metin parçasını sırayla işle
        for text in sample_text_chunks:
            chunks = processor.process_text_chunk(text)
            total_chunks += len(chunks)
        
        # Tüm parçalar oluşturuldu mu?
        assert total_chunks > 0
        assert processor.chunk_count == total_chunks
    
    def test_finalize(self, temp_output_path, sample_text):
        """İşlemi tamamlama testi"""
        processor = TextStreamProcessor(
            output_path=temp_output_path,
            chunk_size=20,
            overlap=5
        )
        
        # Metni işle
        processor.process_text_chunk(sample_text)
        
        # Ek meta veriler
        metadata = {
            "document_id": "test_doc_123",
            "source": "test_source",
            "category": "test"
        }
        
        # İşlemi tamamla
        total_chunks = processor.finalize(metadata)
        
        # Doğru sayıda parça oluşturuldu mu?
        assert total_chunks > 0
        assert total_chunks == processor.chunk_count
        
        # Çıktı dosyası tam mı?
        with open(temp_output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Parçalar var mı?
            assert "chunks" in data
            assert len(data["chunks"]) == total_chunks
            
            # Meta veriler eklendi mi?
            assert "document_id" in data
            assert data["document_id"] == "test_doc_123"
            assert "total_chunks" in data
            assert data["total_chunks"] == total_chunks
    
    def test_empty_text(self, temp_output_path):
        """Boş metin işleme testi"""
        processor = TextStreamProcessor(
            output_path=temp_output_path,
            chunk_size=20,
            overlap=5
        )
        
        # Boş metin işle
        chunks = processor.process_text_chunk("")
        
        # Parça oluşturulmamalı
        assert len(chunks) == 0
        
        # İşlemi tamamla
        total_chunks = processor.finalize()
        
        # Boş metin için parça oluşturulmamalı
        assert total_chunks == 0
        
        # Çıktı dosyası oluşturuldu mu?
        assert os.path.exists(temp_output_path)
        
        # Dosya içeriği doğru mu?
        with open(temp_output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert "chunks" in data
            assert len(data["chunks"]) == 0
    
    def test_remaining_buffer(self, temp_output_path):
        """Tamponda kalan metni işleme testi"""
        processor = TextStreamProcessor(
            output_path=temp_output_path,
            chunk_size=20,
            overlap=5
        )
        
        # Tam chunk_size'dan küçük metin
        test_text = "Bu bir test metnidir. " * 15  # 15 kelime (chunk_size=20'den küçük)
        
        # Metni işle
        chunks = processor.process_text_chunk(test_text)
        
        # Yeterli kelime olmadığından parça oluşturulmamalı
        assert len(chunks) == 0
        
        # İşlemi tamamla
        total_chunks = processor.finalize()
        
        # Tamponda kalan metin için bir parça oluşturulmalı
        assert total_chunks == 1
        
        # Çıktı dosyasını kontrol et
        with open(temp_output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert len(data["chunks"]) == 1
            
            # Son parçanın içeriği doğru mu?
            last_chunk = data["chunks"][0]
            assert last_chunk["text"] == test_text.strip()
            assert last_chunk["length"] == 15
    
    def test_statistics(self, temp_output_path, sample_text):
        """İstatistik toplama testi"""
        processor = TextStreamProcessor(
            output_path=temp_output_path,
            chunk_size=20,
            overlap=5
        )
        
        # Metni işle
        processor.process_text_chunk(sample_text)
        
        # İşlemi tamamla
        processor.finalize()
        
        # Toplam kelime sayısı doğru mu?
        expected_word_count = len(sample_text.split())
        assert processor.total_words == expected_word_count
        
        # Çıktı dosyasındaki istatistikler doğru mu?
        with open(temp_output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert data["total_words"] == expected_word_count
            assert data["chunk_size"] == 20
            assert data["overlap"] == 5
            assert data["total_chunks"] == processor.chunk_count 