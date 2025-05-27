# 🚀 COLAB HIZLI BAŞLANGIÇ REHBERİ

Colab'da proje başlatamıyorsanız, bu adımları sırayla takip edin:

## 🚨 ACİL ÇÖZÜM (2 dakika)

Colab'da yeni bir cell açın ve çalıştırın:

```python
!python emergency_setup.py
```

Bu komut:
- ✅ Temel paketleri yükler
- ✅ Minimal bir uygulama başlatır  
- ✅ PDF yükleme imkanı sağlar
- ✅ Sistem durumunu gösterir

## 🔍 SORUN TESPİTİ

```python
!python colab_debug.py
```

Bu komut tüm sisteminizi analiz eder ve sorunları tespit eder.

## 📋 ADIM ADIM ÇÖZÜM

### 1. Runtime Ayarları (Pro Plus)
```
Runtime > Change runtime type
- Hardware accelerator: GPU 
- Runtime shape: High-RAM
```

### 2. Proje Klonlama
```python
# GitHub'dan klonlayın
!git clone [your-repo-url]
%cd kredi_rag_sistemi

# Dosyaları kontrol edin
!ls -la
```

### 3. Temel Paket Kurulumu
```python
# Python paketlerini güncelleyin
!pip install --upgrade pip

# Temel paketleri kurun
!pip install streamlit pyngrok psutil pandas numpy
```

### 4. AI Paketleri
```python
# AI/ML paketlerini kurun
!pip install sentence-transformers langchain transformers
!pip install faiss-cpu PyPDF2 pdfplumber
```

### 5. Sistem Bağımlılıkları
```python
# Sistem paketlerini kurun
!apt-get update -qq
!apt-get install -y tesseract-ocr tesseract-ocr-tur poppler-utils default-jre
```

### 6. Tam Kurulum
```python
# Pro Plus için
!pip install -r requirements_colab_pro.txt

# veya Free tier için
!pip install -r requirements_colab.txt
```

### 7. Test
```python
!python quick_test_colab.py
```

### 8. Uygulama Başlatma
```python
!python run_colab.py
```

## ⚡ HIZLI KOMUTLAR

### Problem Varsa:
```python
# 1. Runtime'ı yeniden başlatın
# Runtime > Restart runtime

# 2. Acil kurulum
!python emergency_setup.py

# 3. Debug çalıştırın
!python colab_debug.py
```

### PDF Yükleme:
```python
from google.colab import files
import shutil

# PDF'leri yükle
uploaded = files.upload()

# test_pdfs klasörüne taşı
for filename in uploaded.keys():
    shutil.move(filename, f"test_pdfs/{filename}")
    print(f"✅ {filename} yüklendi")
```

### Manuel Streamlit:
```python
# Elle başlatma
!streamlit run streamlit_colab.py --server.port=8501 &

# Ngrok tunnel
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(f"🌐 App URL: {public_url}")
```

## 🛠️ YAYGIN HATALAR VE ÇÖZÜMLERİ

### Hata: "ModuleNotFoundError"
```python
!pip install [eksik-paket-adı]
```

### Hata: "CUDA out of memory"
```python
# CPU moduna geç
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

### Hata: "Streamlit not found"
```python
!pip install streamlit>=1.28.0
```

### Hata: "ngrok tunnel failed"
```python
# Ngrok token ayarla
import os
os.environ["NGROK_AUTH_TOKEN"] = "your_token_here"
```

### Hata: "Permission denied"
```python
# Dosya izinlerini düzelt
!chmod +x *.py
```

## 🎯 PRO PLUS KULLANICILARI İÇİN

Pro Plus aboneliğiniz varsa:

1. **High-RAM Runtime** seçin
2. **GPU** etkinleştirin
3. Bu komutları çalıştırın:

```python
# Pro Plus optimizasyonları
!python colab_setup.py
!python quick_test_colab.py
!python run_colab.py
```

## 📞 DESTEK

Hala sorun yaşıyorsanız:

1. `!python colab_debug.py` çıktısını paylaşın
2. Hata mesajlarını kopyalayın
3. Colab runtime bilgilerinizi kontrol edin

```python
# Sistem bilgileri
import sys, psutil, subprocess
print(f"Python: {sys.version}")
print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB")

try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print("GPU: Available" if result.returncode == 0 else "GPU: Not available")
except:
    print("GPU: Not available")
```

## ✅ BAŞARILI KURULUM KONTROL LİSTESİ

- [ ] Colab Pro Plus runtime seçildi
- [ ] Temel paketler yüklendi
- [ ] AI paketleri yüklendi
- [ ] Proje dosyaları mevcut
- [ ] Test PDF'leri yüklendi
- [ ] Debug testi başarılı
- [ ] Streamlit uygulaması çalışıyor
- [ ] Ngrok tunnel aktif

**🎉 Tebrikler! Sisteminiz çalışıyor.** 