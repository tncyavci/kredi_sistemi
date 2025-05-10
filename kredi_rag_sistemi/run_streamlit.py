import os
import sys
import subprocess

# Ana dizini ekle
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

def main():
    """Streamlit uygulamasını çalıştırır"""
    print("Kredi RAG Streamlit Uygulaması başlatılıyor...")
    
    streamlit_file = os.path.join(ROOT_DIR, "app", "streamlit_app.py")
    
    # Streamlit uygulamasını çalıştır
    subprocess.run([
        "streamlit", "run", 
        streamlit_file,
        "--server.port=8501",
        "--browser.serverAddress=localhost",
        "--server.headless=false"
    ])

if __name__ == "__main__":
    main() 