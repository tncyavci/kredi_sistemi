# Kredi RAG Sistemi Test Dokümantasyonu

Bu dizin, Kredi RAG Sistemi için test dosyalarını içerir. Testler, sistemin doğru çalıştığını doğrulamak ve gelecekteki değişikliklerin mevcut işlevselliği bozmadığından emin olmak için kullanılır.

## Test Kapsamı

Test paketi aşağıdaki test türlerini içerir:

1. **Birim Testleri (Unit Tests)**: Tek bileşenlerin işlevselliğini izole olarak test eder
2. **Entegrasyon Testleri (Integration Tests)**: Birden fazla bileşenin birlikte nasıl çalıştığını test eder
3. **Uçtan Uca Testler (End-to-End Tests)**: Tüm sistemin birlikte çalışabilirliğini test eder
4. **Performans Testleri (Performance Tests)**: Sistemin performansını ve ölçeklenebilirliğini değerlendirir

## Test Çalıştırma Talimatları

### Kurulum

Testleri çalıştırmadan önce gerekli bağımlılıkları yüklediğinizden emin olun:

```bash
pip install -r requirements-dev.txt
```

### Tüm Testleri Çalıştırma

Tüm testleri çalıştırmak için proje kök dizininden:

```bash
pytest kredi_rag_sistemi/tests/
```

### Belirli Test Gruplarını Çalıştırma

Sadece birim testlerini çalıştırma:

```bash
pytest kredi_rag_sistemi/tests/ -m "unit"
```

Sadece entegrasyon testlerini çalıştırma:

```bash
pytest kredi_rag_sistemi/tests/ -m "integration"
```

Uçtan uca testleri çalıştırma:

```bash
pytest kredi_rag_sistemi/tests/ -m "e2e"
```

Performans testlerini çalıştırma (yavaş testler olduğunu unutmayın):

```bash
pytest kredi_rag_sistemi/tests/ -m "performance" --run-performance
```

### Test Raporları

Daha detaylı test raporu almak için:

```bash
pytest kredi_rag_sistemi/tests/ --verbose
```

Kapsam raporu oluşturmak için:

```bash
pytest kredi_rag_sistemi/tests/ --cov=kredi_rag_sistemi
```

HTML kapsam raporu oluşturmak için:

```bash
pytest kredi_rag_sistemi/tests/ --cov=kredi_rag_sistemi --cov-report=html
```

## Test Dosyaları

Test dizinindeki ana dosyalar şunlardır:

- `conftest.py`: Ortak pytest yapılandırması ve fixture'ları
- `create_test_pdfs.py`: Test için örnek PDF dosyaları oluşturur
- `test_pdf_processor.py`: Temel PDF işleme işlevselliği testleri
- `test_enhanced_pdf_processor.py`: Gelişmiş PDF işleme işlevselliği birim testleri
- `test_integration_pdf_processor.py`: PDF işleme entegrasyon testleri
- `test_performance_pdf_processor.py`: PDF işleme performans testleri
- `test_end_to_end_pdf_processor.py`: PDF işleme uçtan uca testleri
- `test_pdf_processor_compatibility.py`: Eski ve yeni PDF işleyiciler arasındaki uyumluluk testleri
- `test_text_stream_processor.py`: Metin akış işleme modülü testleri

## Test PDF Dosyaları

Testler, `kredi_rag_sistemi/test_pdfs/` dizininde bulunan test PDF dosyalarını kullanır. Test çalıştırıldığında test PDF dosyaları otomatik olarak oluşturulur, ancak gerekirse manuel olarak da oluşturabilirsiniz:

```bash
python kredi_rag_sistemi/tests/create_test_pdfs.py
```

## CI/CD Entegrasyonu

Testler, Sürekli Entegrasyon (CI) süreçlerine entegre edilebilir. GitHub Actions, GitLab CI, Jenkins vb. için örnek yapılandırmalar `ci/` dizininde bulunabilir.

## Test Geliştirme

Yeni testler eklerken aşağıdaki kurallara uyulmalıdır:

1. Uygun test markerlarını kullanın (`@pytest.mark.unit`, `@pytest.mark.integration`, vb.)
2. Dokümante edilmiş fixture'ları mümkün olduğunca yeniden kullanın
3. Her test için açıklayıcı docstring ekleyin
4. Test fonksiyon adları `test_` ile başlamalıdır

## Hata Ayıklama

Test hataları durumunda daha fazla bilgi için test loglarını kontrol edin:

```
kredi_rag_sistemi/tests/logs/test_run.log
``` 