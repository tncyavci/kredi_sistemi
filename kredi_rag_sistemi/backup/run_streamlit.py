#!/usr/bin/env python3
"""
Kredi RAG Sistemi - CPU Optimized Streamlit Launcher
"""
import os
import sys
import subprocess

# CPU-only environment setup
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["FORCE_CPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ana dizini ekle
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

def main():
    """Streamlit uygulamasını CPU-optimized modda çalıştırır"""
    print("🚀 Kredi RAG Sistemi - CPU Optimized Mode")
    print("=" * 50)
    
    streamlit_file = os.path.join(ROOT_DIR, "app", "streamlit_app.py")
    
    if not os.path.exists(streamlit_file):
        print(f"❌ Streamlit app bulunamadı: {streamlit_file}")
        sys.exit(1)
    
    print("🌐 Streamlit uygulaması başlatılıyor...")
    print("📱 Tarayıcınızda otomatik olarak açılacak")
    print("🔗 Manuel açmak için: http://localhost:8501")
    print("🛑 Durdurmak için: Ctrl+C")
    print("=" * 50)
    
    # Streamlit uygulamasını çalıştır
    try:
        subprocess.run([
            "streamlit", "run", 
            streamlit_file,
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false",
            "--theme.base=light"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Uygulama durduruldu.")
    except Exception as e:
        print(f"❌ Başlatma hatası: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 