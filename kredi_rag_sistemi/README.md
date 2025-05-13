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

## Geliştirilmiş PDF İşleme Özellikleri

- **GPU Destekli OCR**: EasyOCR ile GPU destekli, hızlı metin tanıma
- **Önbellekleme Sistemi**: Daha önce işlenmiş PDF'leri yeniden işlemeden kullanma
- **Daha Yüksek Doğruluk**: Çoklu OCR motorlarını birleştirerek daha doğru sonuçlar
- **Büyük Dosya Desteği**: 500MB'a kadar büyük PDF'leri işleyebilme
- **Tablo Çıkarım İyileştirmeleri**: Farklı tablo yapılarını daha iyi tanıyabilme
- **Çoklu PDF Formatları**: Şifreli olmayan, taranmış ve melez PDF'leri destekleme
- **Paralel İşleme**: Çoklu çekirdek ve threading ile hızlı işleme
- **Performans İzleme**: İşleme zamanı ve kullanılan kaynakları izleme

## Kurulum

```bash
# Gerekli kütüphaneleri kur
pip install -r requirements.txt

# Tesseract OCR kurulumu (OS'a göre değişir)
# Ubuntu:
sudo apt-get install tesseract-ocr
# macOS:
brew install tesseract

# Tensorflow/PyTorch GPU desteği için ekstra adımlar gerekebilir
```

## Kullanım

1. PDF'leri işleme:

```bash
python optimize_pdf_process.py --use_gpu --chunk_size 3000 --overlap 300
```

2. RAG sorgulama:

```bash
python app/main.py
```

3. API ve UI'yi başlatma:

```bash
uvicorn app.api:app --reload
streamlit run app/ui.py
```

## Performans İzleme ve Test

Projenin performansını test etmek için:

```bash
python -m pytest tests/test_pdf_processing_performance.py -v
```

Bu, farklı senaryolarda (CPU/GPU, önbellekli/önbelleksiz) performans karşılaştırması yaparak en iyi konfigürasyonu belirlemenize yardımcı olur.

## Lisans

MIT

## Katkıda Bulunanlar

- Örnek Katkıcı 