# Kredi RAG Sistemi

Bu proje, kredi başvuruları ve finansal belgeler için geliştirilmiş Retrieval-Augmented Generation (RAG) tabanlı bir bilgi erişim sistemidir. Sistem, PDF formatındaki finansal belgeleri işleyerek kullanıcıların Türkçe doğal dil sorguları ile belgelerdeki bilgilere kolayca erişmesini sağlar.

## 🚀 Özellikler

- PDF belgelerinden metin çıkarma ve işleme
- Metin parçalama ve çok dilli vektör dönüşümü
- ChromaDB veya FAISS ile güvenli ve hızlı vektör saklama
- Yerel Mistral 7B modeli ile doğal dil yanıtları
- Streamlit tabanlı kullanıcı dostu arayüz
- FastAPI ile RESTful API desteği
- Docker ile kolay dağıtım
- Merkezi loglama ve izleme
- Güvenli kimlik doğrulama ve API erişimi

## 📋 Kurulum

### Gereksinimler

- Python 3.10 veya üzeri
- Poppler (PDF işleme için)
- Tesseract OCR (isteğe bağlı, OCR için)

### 1. Bağımlılıkları Yükleme

```bash
# Temel bağımlılıklar
pip install -r requirements.txt

# Geliştirme bağımlılıkları (opsiyonel)
pip install -r requirements-dev.txt
```

### 2. Model İndirme

```bash
# Mistral 7B modelini indir
python -c "from models.llm import download_mistral_model; download_mistral_model()"
```

## 🔧 Kullanım

### Streamlit Uygulaması

Etkileşimli web arayüzünü başlatmak için:

```bash
python run_streamlit.py
```

Bu komut, `http://localhost:8501` adresinde Streamlit arayüzünü başlatacaktır.

### API Servisi

RESTful API'yi başlatmak için:

```bash
python run_api.py
```

API, `http://localhost:8000` adresinde çalışmaya başlayacaktır. API dökümantasyonu `http://localhost:8000/docs` adresinde bulunabilir.

### PDF İşleme

PDF belgelerini komut satırından işlemek için:

```bash
python process_pdfs.py
```

### Docker ile Çalıştırma

```bash
# Tek servis
docker build -t kredi-rag .
docker run -p 8501:8501 kredi-rag

# veya Docker Compose ile tüm hizmetleri başlatma
docker-compose up -d
```

## 📁 Proje Yapısı

```
kredi_rag_sistemi/
├── app/                     # Uygulama kodları
│   ├── api/                 # FastAPI uygulaması
│   │   ├── app.py           # API başlatıcı
│   │   └── router.py        # API rotaları
│   ├── core/                # Temel işlevler
│   │   ├── rag.py           # Ana RAG sınıfı
│   │   └── pdf_processor.py # PDF işleme
│   ├── services/            # Servis modülleri
│   ├── streamlit_app.py     # Streamlit arayüzü
│   ├── utils/               # Yardımcı fonksiyonlar
│   │   └── rag_utils.py     # RAG yardımcıları
│   ├── security.py          # Güvenlik ayarları
│   └── config.py            # Konfigürasyon
├── models/                  # Model tanımları
│   ├── llm.py               # LLM entegrasyonu
│   ├── embeddings.py        # Embedding modelleri
│   └── vector_store.py      # Vektör depolama
├── utils/                   # Genel yardımcılar
│   └── logging_config.py    # Loglama ayarları
├── tests/                   # Test dosyaları
├── data/                    # Veri dosyaları
│   ├── processed/           # İşlenmiş veriler
│   └── raw/                 # Ham veriler
├── logs/                    # Log dosyaları
├── test_pdfs/               # Test PDF'leri
├── docker-compose.yml       # Docker Compose
├── Dockerfile               # Docker yapılandırması
├── run_streamlit.py         # Streamlit başlatıcı
├── run_api.py               # API başlatıcı
├── process_pdfs.py          # PDF işleme betiği
├── requirements.txt         # Bağımlılıklar
├── requirements-dev.txt     # Geliştirme bağımlılıkları
└── README.md                # Bu doküman
```

## ⚙️ Konfigürasyon

Sistem ayarlarını `app/config.py` dosyasından özelleştirebilirsiniz:

- Model ayarları (embedding modeli, LLM parametreleri)
- PDF işleme ayarları (parça boyutu, örtüşme)
- Vektör veritabanı ayarları (ChromaDB/FAISS)
- Kategori eşleştirmeleri
- Sistem promptu

## 🔍 Vektör Veritabanı Seçimi

Sistem iki farklı vektör veritabanını destekler:

### ChromaDB
- Kolay kullanım ve kurulum
- Zengin metaveri desteği
- Orta düzeyde belge koleksiyonları için ideal

### FAISS (Facebook AI Similarity Search)
- Yüksek performans
- Büyük ölçekli veri setleri için optimize edilmiş
- Gelişmiş arama yetenekleri

`app/config.py` dosyasındaki `VECTOR_DB_CONFIG` ayarından seçiminizi yapabilirsiniz.

## 🧪 Test

Testleri çalıştırmak için:

```bash
# Tüm testler
pytest

# Spesifik test dosyası
pytest tests/test_rag_system.py
```

## 🔒 Güvenlik

- PDF dosya boyutu sınırlaması
- Dosya uzantısı kontrolü
- Güvenli vektör veritabanı saklama
- API güvenliği için temel kimlik doğrulama
- Opsiyonel veri şifreleme
- `.env` dosyası ile hassas bilgilerin yönetimi

## 🤝 Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 📬 İletişim

Sorularınız için bu repo üzerinden iletişime geçebilirsiniz. 