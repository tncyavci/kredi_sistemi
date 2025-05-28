# 🏦 Kredi RAG Sistemi v2.0

Modern, temiz mimari ile geliştirilmiş Retrieval-Augmented Generation tabanlı finansal belge analiz sistemi.

## 🚀 Hızlı Başlangıç

### Web Interface
```bash
python run.py web
```

### API Interface  
```bash
python run.py api
```

### Performance Testi
```bash
python run.py optimize
```

## 📁 Proje Yapısı

```
kredi_rag_sistemi/
├── 📂 src/                     # Ana kaynak kod
│   ├── 📂 core/               # Business logic
│   │   ├── rag_engine.py     # RAG motoru
│   │   └── document_processor.py # PDF işleme
│   ├── 📂 models/             # Model interface'leri
│   │   ├── llm_interface.py  # LLM abstraction
│   │   ├── vector_store.py   # Vector DB
│   │   └── embeddings.py     # Embedding models
│   ├── 📂 services/           # Business services
│   └── 📂 utils/             # Utility fonksiyonlar
│
├── 📂 interfaces/             # User interfaces
│   ├── 📂 web/               # Streamlit web UI
│   └── 📂 api/               # REST API
│
├── 📂 tests/                  # Test dosyaları
│   ├── 📂 unit/              # Unit testler
│   ├── 📂 integration/       # Integration testler
│   ├── 📂 performance/       # Performance testler
│   └── 📂 debug/            # Debug scriptleri
│
├── 📂 scripts/               # Utility scriptler
├── 📂 configs/               # Konfigürasyon
├── 📂 docs/                  # Dokümantasyon
└── 📂 docker/               # Container dosyaları
```

## 🔧 Kurulum

1. **Python Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# veya
.venv\Scripts\activate     # Windows
```

2. **Dependencies**
```bash
pip install -r requirements.txt
```

3. **Environment Setup**
```bash
cp .env.example .env
# .env dosyasını düzenleyin
```

## 🌟 Özellikler

### ⚡ Performance Optimizations
- **Session State Management**: RAG instance'ı bir kez yüklenir
- **Query Caching**: 5 dakika TTL ile sorgu cache'i
- **Duplicate Filtering**: Belge tekrarlarını önler
- **Memory Optimization**: %70 memory kullanım azalışı

### 🧠 AI Capabilities
- **Local LLM**: Mistral 7B CPU-optimized
- **Vector Search**: Semantic similarity arama
- **Financial Focus**: Finansal belge analiz odaklı
- **Turkish Support**: Türkçe dil desteği

### 🔐 Enterprise Ready
- **Clean Architecture**: Modüler, sürdürülebilir yapı
- **Configuration Management**: YAML tabanlı config
- **Logging**: Comprehensive logging sistemi
- **Docker Support**: Container deployment

## 📊 Performance Metrics

| Metrik | Önceki Sistem | Yeni Sistem | İyileştirme |
|--------|---------------|-------------|-------------|
| İlk sorgu | ~25 saniye | ~5 saniye | %80 |
| Cache hit | N/A | ~1 saniye | %95 |
| Memory kullanımı | Yüksek | Sabit | %70 azalış |
| Model yeniden yükleme | Her sorgu | Hiç | %100 |

## 🧪 Test Etme

### Tüm testler
```bash
python run.py test
```

### Performance testi
```bash
python run.py optimize
```

### Debug scriptleri
```bash
cd tests/debug
python check_vector_db.py
```

## 🐳 Docker Deployment

```bash
cd docker
docker-compose up -d
```

## 📚 Dokümantasyon

- [Optimization Guide](docs/OPTIMIZATION_GUIDE.md)
- [Architecture Design](docs/project_restructure_plan.md)
- [API Documentation](docs/api_documentation.md)

## 🛠️ Geliştirme

### Code Standards
- Clean Architecture principles
- Type hints kullanımı
- Comprehensive error handling
- Performance-first approach

### Contributing
1. Feature branch oluştur
2. Testleri çalıştır
3. Performance impact kontrol et
4. Pull request oluştur

## 🚧 Roadmap

- [ ] **Batch Processing**: Multiple query processing
- [ ] **Model Quantization**: Smaller model sizes
- [ ] **Async Support**: Non-blocking operations
- [ ] **Advanced Caching**: Redis integration
- [ ] **Monitoring**: Prometheus metrics
- [ ] **Auto-scaling**: Kubernetes deployment

## 📞 Destek

Sorunlar için GitHub Issues kullanın veya sistem loglarını kontrol edin:

```bash
tail -f logs/streamlit.log
```

---

**Kredi RAG Sistemi v2.0** - Modern AI ile finansal belge analizi 