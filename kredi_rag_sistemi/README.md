# Kredi RAG Sistemi

Bu proje, kredi başvuruları ve finansal belgeler için Retrieval-Augmented Generation (RAG) tabanlı bir soru-cevap sistemi sunar. Sistem, PDF formatındaki finansal belgeleri işleyerek, kullanıcıların doğal dil kullanarak belgelerdeki bilgilere erişmesini sağlar.

## Özellikler

- PDF belgelerinden metin çıkarma ve işleme
- Metin parçalama ve vektörleştirme
- Benzerlik tabanlı belge arama
- Yerel Mistral 7B modeli ile doğal dil işleme
- Etkileşimli sorgu arayüzü

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Mistral modelini indirin:
```bash
python -c "from models.llm import download_mistral_model; download_mistral_model()"
```

## Kullanım

### PDF İşleme

PDF dosyalarını işlemek için:
```bash
python app/services/pdf_rag_demo.py --pdf_dir /path/to/pdfs
```

### Etkileşimli Mod

Etkileşimli sorgu modunu başlatmak için:
```bash
python app/services/pdf_rag_demo.py --pdf_dir /path/to/pdfs --interactive
```

## Yapı

```
kredi_rag_sistemi/
├── app/
│   ├── core/
│   │   └── rag.py           # Ana RAG sınıfı
│   ├── services/
│   │   └── pdf_rag_demo.py  # Demo uygulaması
│   ├── utils/
│   │   └── rag_utils.py     # Yardımcı fonksiyonlar
│   └── config.py            # Konfigürasyon ayarları
├── models/
│   ├── llm.py              # LLM modeli
│   └── embeddings.py       # Embedding modeli
├── utils/
│   └── preprocessing/
│       └── pdf_processor.py # PDF işleme
├── tests/                  # Test dosyaları
├── data/                   # Veri dosyaları
├── logs/                   # Log dosyaları
└── requirements.txt        # Bağımlılıklar
```

## Konfigürasyon

Sistem ayarlarını `app/config.py` dosyasından özelleştirebilirsiniz:

- Model ayarları (embedding modeli, LLM parametreleri)
- PDF işleme ayarları (parça boyutu, örtüşme)
- Vektör veritabanı ayarları
- Kategori eşleştirmeleri
- Sistem promptu

## Test

Testleri çalıştırmak için:
```bash
pytest tests/
```

## Performans İyileştirmeleri

- Bellek yönetimi için düzenli GC çağrıları
- Büyük belgeler için parçalı işleme
- FAISS ile hızlı benzerlik araması
- Paralel PDF işleme desteği

## Güvenlik

- PDF dosya boyutu sınırlaması
- Dosya uzantısı kontrolü
- Güvenli vektör veritabanı saklama

## Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## İletişim

Sorularınız veya önerileriniz için:
- E-posta: [e-posta adresi]
- GitHub Issues: [GitHub Issues sayfası] 