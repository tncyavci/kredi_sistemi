FROM python:3.10-slim

WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    poppler-utils \
    libpoppler-cpp-dev \
    tesseract-ocr \
    tesseract-ocr-tur \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Çoklu Mimari Desteği için LLAMA_CPP_PYTHON optimizasyon bayrağı
ENV LLAMA_CPP_PYTHON_FORCE_CMAKE=1

# Projeyi kopyala
COPY . .

# Gerekli dizinleri oluştur
RUN mkdir -p logs data/processed data/raw models/embeddings

# Uygulama için port aç (FastAPI için 8000, Streamlit için 8501)
EXPOSE 8000 8501

# Model indirme betiği çalıştır
RUN python -c "from models.llm import download_mistral_model; download_mistral_model('./models')"

# Çalıştırma komutu - varsayılan olarak Streamlit
CMD ["python", "run_streamlit.py"] 