# 🏦 Kredi RAG Sistemi - Google Colab Kullanım Kılavuzu

Bu kılavuz, Kredi RAG Sistemini Google Colab'da nasıl çalıştıracağınızı detaylı olarak açıklar.

## 🚀 Hızlı Başlangıç

### 1. Projeyi Google Colab'a Yükleme

Google Colab'da yeni bir notebook açın ve aşağıdaki kodları çalıştırın:

```python
# Projeyi GitHub'dan klonlayın (kendi repository URL'nizi kullanın)
!git clone https://github.com/[your-username]/kredi_rag_sistemi.git
%cd kredi_rag_sistemi

# Proje dosyalarını listeleyin
!ls -la
```

### 2. Sistem Kurulumu

```python
# Kurulum scriptini çalıştırın
!python colab_setup.py
```

Bu script:
- Sistem bağımlılıklarını yükler (Tesseract, Java, vb.)
- Python paketlerini yükler
- Gerekli dizinleri oluşturur
- Colab ortamını konfigüre eder

### 3. Uygulamayı Başlatma

```python
# Streamlit uygulamasını ngrok tunnel ile başlatın
!python run_colab.py
```

Bu komut çalıştıktan sonra size bir ngrok URL'i verilecek. Bu URL'ye tıklayarak uygulamanıza erişebilirsiniz.

## 📄 PDF Dosyalarını Yükleme

### Yöntem 1: Doğrudan Colab'da Yükleme

```python
from google.colab import files
import shutil
import os

# PDF dosyalarını yükleyin
uploaded = files.upload()

# Yüklenen dosyaları test_pdfs dizinine taşıyın
os.makedirs("test_pdfs", exist_ok=True)
for filename in uploaded.keys():
    shutil.move(filename, f"test_pdfs/{filename}")
    print(f"✅ {filename} test_pdfs dizinine taşındı")
```

### Yöntem 2: Google Drive'dan Yükleme

```python
from google.colab import drive
import shutil

# Google Drive'ı bağlayın
drive.mount('/content/drive')

# Drive'dan PDF dosyalarını kopyalayın
source_folder = "/content/drive/MyDrive/PDFs"  # Drive'daki PDF klasörünüz
target_folder = "test_pdfs"

if os.path.exists(source_folder):
    for filename in os.listdir(source_folder):
        if filename.endswith('.pdf'):
            shutil.copy2(f"{source_folder}/{filename}", f"{target_folder}/{filename}")
            print(f"✅ {filename} kopyalandı")
```

## 🔧 Konfigürasyon ve Optimizasyon

### Bellek Yönetimi

Google Colab'da bellek sınırlı olduğu için:

```python
import gc
import psutil

# Bellek kullanımını kontrol edin
def check_memory():
    memory = psutil.virtual_memory()
    print(f"Toplam RAM: {memory.total / (1024**3):.1f}GB")
    print(f"Kullanılan: {memory.percent:.1f}%")
    print(f"Mevcut: {memory.available / (1024**3):.1f}GB")

check_memory()

# Bellek temizleme
gc.collect()
```

### GPU vs CPU Kullanımı

```python
# GPU durumunu kontrol edin
!nvidia-smi

# GPU kullanımını aktifleştirmek için (opsiyonel)
import os
os.environ["COLAB_ENV"] = "1"
os.environ["USE_GPU"] = "1"  # GPU varsa kullan
```

## 📊 Sistem Sınırlamaları ve Öneriler

### Google Colab Pro Plus Avantajları:
- **RAM**: 52GB'a kadar (High-RAM runtime)
- **GPU**: T4, V100, A100 erişimi
- **Disk**: ~100GB geçici
- **Session**: 24+ saat timeout
- **Priority**: Hızlı GPU erişimi

### Google Colab Free Tier Sınırları:
- **RAM**: ~12GB
- **Disk**: ~50GB geçici  
- **Session**: ~12 saat timeout
- **GPU**: T4 (sınırlı süre)

### Pro Plus Optimizasyon Önerileri:

1. **Dosya Boyutları (Pro Plus)**:
   - Maksimum PDF boyutu: 100MB
   - Maksimum sayfa sayısı: 200 sayfa/PDF
   - Toplam işlem boyutu: 500MB
   - Paralel işleme destegi

2. **Free Tier Ayarları**:
   - Maksimum PDF boyutu: 25MB
   - Maksimum sayfa sayısı: 50 sayfa/PDF
   - Toplam işlem boyutu: 100MB

3. **Bellek Yönetimi**:
   - Pro Plus: Yüksek bellek threshold (4GB)
   - Free Tier: Düşük threshold (1GB)
   - Otomatik cache yönetimi

4. **Processing Settings**:
   - Pro Plus: GPU acceleration + CPU paralel işleme
   - Free Tier: CPU-only modda çalışma
   - Adaptif chunk boyutları

## 🛠️ Sorun Giderme

### Yaygın Problemler ve Çözümleri:

#### 1. "Out of Memory" Hatası
```python
# Runtime'ı yeniden başlatın
# Menü: Runtime > Restart runtime

# Daha sonra sadece gerekli paketleri yükleyin
!pip install streamlit langchain sentence-transformers faiss-cpu

# Bellek kullanımını azaltın
import gc
gc.collect()
```

#### 2. Ngrok Tunnel Problemi
```python
# Ngrok auth token ayarlayın (https://ngrok.com/)
import os
os.environ["NGROK_AUTH_TOKEN"] = "your_token_here"

# Tunnel'ları temizleyin
!pkill -f ngrok
```

#### 3. PDF İşleme Hatası
```python
# Tesseract'ın doğru yüklü olduğunu kontrol edin
!tesseract --version

# Java'nın yüklü olduğunu kontrol edin (tabula için)
!java -version

# PDF dosyasının boyutunu kontrol edin
import os
for file in os.listdir("test_pdfs"):
    if file.endswith('.pdf'):
        size_mb = os.path.getsize(f"test_pdfs/{file}") / (1024*1024)
        print(f"{file}: {size_mb:.1f}MB")
```

#### 4. Model İndirme Problemi
```python
# Internet bağlantısını kontrol edin
!ping -c 3 huggingface.co

# Hugging Face cache'ini temizleyin
!rm -rf ~/.cache/huggingface/

# Manuel model indirme
from huggingface_hub import snapshot_download
snapshot_download(repo_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
```

## 📱 Kullanım İpuçları

### 1. Verimli PDF İşleme
- Küçük dosyalarla test edin
- Önce tek PDF ile deneyin
- Büyük dosyaları sayfalara bölün

### 2. Soru Sorma
- Türkçe sorular sorun
- Spesifik bilgi isteyin
- Tablo içeriği için detaylı sorular

### 3. Performans Optimizasyonu
- Gereksiz dosyaları silin
- Cache'leri kullanın
- Bellek kullanımını takip edin

## 🔄 Güncelleme ve Bakım

### Proje Güncelleme
```python
# En son değişiklikleri çekin
!git pull origin main

# Gereksinimleri güncelleyin
!pip install -r requirements_colab.txt --upgrade
```

### Cache Temizleme
```python
# PDF cache'ini temizleyin
!rm -rf cache/

# Model cache'ini temizleyin
!rm -rf models/embeddings/

# Log dosyalarını temizleyin
!rm -rf logs/
```

## 🆘 Destek ve Yardım

### Hata Raporlama
Sorun yaşadığınızda aşağıdaki bilgileri toplayın:

```python
# Sistem bilgilerini topla
import sys, torch, transformers, langchain

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("LangChain:", langchain.__version__)

# Bellek durumu
import psutil
memory = psutil.virtual_memory()
print(f"RAM: {memory.percent:.1f}% kullanımda")

# GPU durumu
!nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
```

### Performans İzleme
```python
import time
import matplotlib.pyplot as plt

# İşlem sürelerini kaydet
times = []
memory_usage = []

def monitor_performance():
    start_time = time.time()
    # PDF işleme kodunuz burada
    end_time = time.time()
    
    times.append(end_time - start_time)
    memory_usage.append(psutil.virtual_memory().percent)
    
    # Grafik çiz
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(times)
    plt.title('İşlem Süreleri')
    plt.ylabel('Saniye')
    
    plt.subplot(1, 2, 2)
    plt.plot(memory_usage)
    plt.title('Bellek Kullanımı')
    plt.ylabel('%')
    
    plt.tight_layout()
    plt.show()
```

---

## ⚡ Hızlı Başlangıç Özeti

### Pro Plus Kullanıcıları İçin:
1. `!git clone [repo-url] && cd kredi_rag_sistemi`
2. **High-RAM runtime** seçin (Colab > Runtime > Change runtime type)
3. `!python colab_setup.py`  # Pro Plus optimizasyonları otomatik algılanır
4. `!python quick_test_colab.py`  # Sistem testi (opsiyonel)
5. `!python run_colab.py`
6. 100MB'a kadar PDF'leri yükleyin ve işleyin
7. Gelişmiş Türkçe sorular sorun

### Free Tier Kullanıcıları İçin:
1. `!git clone [repo-url] && cd kredi_rag_sistemi`
2. `!python colab_setup.py`
3. `!python run_colab.py`  
4. 25MB'a kadar PDF'leri yükleyin ve işleyin
5. Türkçe sorular sorun

### 🎯 Pro Plus Avantajları:
- 🚀 2-5x daha hızlı işleme
- 📊 4x daha büyük dosya desteği
- 🎯 GPU acceleration
- 💾 52GB'a kadar RAM
- 🔄 Paralel işleme
- ⏰ 24+ saat session

**🎉 Başarılar! Artık Google Colab'da güçlü Kredi RAG sisteminiz çalışıyor.** 