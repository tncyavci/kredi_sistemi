# Kredi RAG Sistemi

Bu sistem, finans kurumları için kredi süreçlerini hızlandırmak amacıyla PDF dosyalarından bilgi çıkararak soru-cevap yapabilen bir RAG (Retrieval-Augmented Generation) sistemidir.

## Özellikler

- **PDF İşleme**: Finansal belgeleri, sözleşmeleri ve formları işleyerek bilgi çıkarır
- **OCR Desteği**: Taranan belgelerdeki metni tanıma
- **Tablo Tanıma**: Finansal tablolardaki verileri yapılandırılmış biçimde çıkarma
- **GPU Destekli İşleme**: Performanslı OCR ve belge işleme için GPU desteği
- **Önbellekleme Sistemi**: İşlenmiş belgeleri önbellekleyerek tekrarlı işlemleri hızlandırma
- **Çoklu Vektör Tabanı**: ChromaDB, Milvus veya Pinecone ile uyumluluk
- **Lokal LLM Entegrasyonu**: Gizlilik gerektiren uygulamalar için lokal LLM desteği
- **Türkçe Sorgu Desteği**: Türkçe finansal terimler ve sorgular için optimizasyon

## Geliştirilmiş PDF İşleme Özellikleri

- **GPU Destekli OCR**: EasyOCR ile GPU destekli, hızlı metin tanıma
- **Önbellekleme Sistemi**: Daha önce işlenmiş PDF'leri yeniden işlemeden kullanma
- **Daha Yüksek Doğruluk**: Çoklu OCR motorlarını birleştirerek daha doğru sonuçlar
- **Büyük Dosya Desteği**: 500MB'a kadar büyük PDF'leri işleyebilme
- **Tablo Çıkarım İyileştirmeleri**: Farklı tablo yapılarını daha iyi tanıyabilme
- **Çoklu PDF Formatları**: Şifreli olmayan, taranmış ve melez PDF'leri destekleme
- **Paralel İşleme**: Çoklu çekirdek ve threading ile hızlı işleme
- **Performans İzleme**: İşleme zamanı ve kullanılan kaynakları izleme
- **Yapılandırılmış Tablolar**: Tablolardaki veri hücreleri arasındaki ilişkileri daha iyi tanıma
- **Veri Normalizasyonu**: Türkçe karakter ve metin normalizasyonu ile daha doğru sonuçlar

## Kurulum

```bash
# Gerekli kütüphaneleri kur
pip install -r requirements.txt

# Tesseract OCR kurulumu (OS'a göre değişir)
# Ubuntu:
sudo apt-get install tesseract-ocr tesseract-ocr-tur  # Türkçe dil paketi ekli
# macOS:
brew install tesseract tesseract-lang  # Tüm diller

# GPU desteği için
pip install easyocr
pip install nest_asyncio  # Streamlit için gerekli
```

## Kullanım

1. PDF'leri işleme:

```bash
python optimize_pdf_process.py --use_gpu --chunk_size 3000 --overlap 300
```

2. Web arayüzünü başlatma:

```bash
streamlit run app/streamlit_app.py
```

3. API'yi başlatma:

```bash
uvicorn app.api:app --reload
```

## Sorgu Örnekleri

Sistem, Türkçe ve İngilizce PDF'leri hem Türkçe hem de İngilizce sorgularla yanıtlayabilir:

- **Türkçe**: "2021 yılı için toplam aktifler ne kadardır?"
- **Türkçe**: "Pegasus'un özel finansal bilgileri nelerdir?"
- **Türkçe**: "Nakit akışı tablosunda en büyük gider kalemi nedir?"
- **İngilizce**: "What is the total assets for 2021?"

## Tablo Anlama Kapasitesi

Sistem, tablolardaki bilgileri anlama konusunda özel olarak optimize edilmiştir:

- Tablo yapılarını algılar ve içindeki ilişkileri anlar
- Hücre değerlerini ve sütun başlıklarını doğru şekilde eşleştirir
- Finansal terimleri ve sayıları anlamlandırabilir
- Tablolar arası karşılaştırma yapabilir

## Performans İzleme ve Test

Projenin performansını test etmek için:

```bash
python -m pytest tests/test_pdf_processing_performance.py -v
```

Bu, farklı senaryolarda (CPU/GPU, önbellekli/önbelleksiz) performans karşılaştırması yaparak en iyi konfigürasyonu belirlemenize yardımcı olur.

## Sorun Giderme

- **OCR Sorunları**: Tesseract veya EasyOCR yüklü değilse, metin çıkarma doğruluğu düşebilir
- **GPU Sorunları**: GPU kullanırken hata alırsanız, `--use_gpu false` parametresiyle CPU modunda çalıştırın
- **Türkçe Karakter Sorunları**: Türkçe karakter sorunu yaşarsanız, `LANG=tr_TR.UTF-8` ortam değişkenini ayarlayın

## Lisans

MIT

## Katkıda Bulunanlar

- Elvin Ertuğrul
- Tuncay Avci 