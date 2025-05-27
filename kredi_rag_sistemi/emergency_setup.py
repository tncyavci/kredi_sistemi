#!/usr/bin/env python3
"""
Acil durum kurulum scripti - Colab'da hızlı çözüm
"""

import os
import sys
import subprocess

def emergency_install():
    """Emergency package installation"""
    print("🚨 Emergency Setup Starting...")
    print("=" * 50)
    
    # Set environment
    os.environ["COLAB_ENV"] = "1"
    os.environ["COLAB_PRO_PLUS"] = "1"
    
    # Essential packages only
    essential_packages = [
        "streamlit>=1.28.0",
        "pyngrok>=5.1.0",
        "psutil>=5.9.0",
        "sentence-transformers>=2.2.2",
        "langchain>=0.1.4",
        "faiss-cpu>=1.7.4",
        "PyPDF2>=3.0.1",
        "pandas>=2.0.3",
        "numpy>=1.24.3"
    ]
    
    print("📦 Installing essential packages...")
    for package in essential_packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-q", package
            ], check=True)
            print(f"✅ {package}")
        except Exception as e:
            print(f"⚠️ Failed to install {package}: {e}")
    
    # Create directories
    directories = ["data/processed", "models/embeddings", "logs", "cache", "test_pdfs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Emergency setup completed!")

def create_minimal_streamlit():
    """Create minimal Streamlit app"""
    minimal_app = '''
import streamlit as st
import os
import sys

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

st.title("🏦 Kredi RAG Sistemi - Minimal Version")
st.success("Emergency setup successful! Basic version is running.")

st.subheader("📋 Next Steps:")
st.write("1. Upload PDF files using the file uploader below")
st.write("2. Run full setup: `!python colab_setup.py`")
st.write("3. Install all packages: `!pip install -r requirements_colab_pro.txt`")

# File uploader
uploaded_files = st.file_uploader(
    "Upload PDF files", 
    type="pdf", 
    accept_multiple_files=True
)

if uploaded_files:
    import os
    os.makedirs("test_pdfs", exist_ok=True)
    
    for uploaded_file in uploaded_files:
        with open(f"test_pdfs/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getvalue())
        st.success(f"✅ {uploaded_file.name} uploaded to test_pdfs/")

st.subheader("🔧 System Info:")
import psutil
memory = psutil.virtual_memory()
st.info(f"RAM: {memory.total / (1024**3):.1f}GB total, {memory.available / (1024**3):.1f}GB available")

st.subheader("📱 Full App Launch:")
st.code("!python run_colab.py")
'''
    
    with open("emergency_app.py", "w") as f:
        f.write(minimal_app)
    
    print("✅ Minimal Streamlit app created: emergency_app.py")

def launch_emergency_app():
    """Launch emergency Streamlit app"""
    try:
        from pyngrok import ngrok
        import subprocess
        import time
        
        # Kill existing tunnels
        ngrok.kill()
        time.sleep(2)
        
        print("🚀 Starting emergency Streamlit app...")
        proc = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run",
            "emergency_app.py",
            "--server.port=8502",
            "--server.headless=true",
            "--browser.gatherUsageStats=false"
        ])
        
        time.sleep(8)
        
        # Create tunnel
        public_url = ngrok.connect(8502)
        
        print("=" * 60)
        print("🎉 EMERGENCY APP RUNNING!")
        print(f"🌐 Emergency URL: {public_url}")
        print("📱 Use this to upload PDFs and check system status")
        print("=" * 60)
        
        return proc, public_url
        
    except Exception as e:
        print(f"❌ Failed to launch emergency app: {e}")
        return None, None

def main():
    """Main emergency setup"""
    print("🚨 EMERGENCY SETUP FOR COLAB")
    print("This will install minimal packages and create a basic app")
    print("=" * 60)
    
    # Emergency install
    emergency_install()
    
    # Create minimal app
    create_minimal_streamlit()
    
    # Try to launch
    proc, url = launch_emergency_app()
    
    if url:
        print("\n✅ Emergency setup successful!")
        print("\n📋 Next steps:")
        print("1. Use the emergency app to upload PDFs")
        print("2. Run: !python colab_debug.py")
        print("3. Run: !python colab_setup.py")
        print("4. Run: !python run_colab.py")
    else:
        print("\n⚠️ Could not launch emergency app")
        print("Try running manually:")
        print("!streamlit run emergency_app.py --server.port=8502")

if __name__ == "__main__":
    main() 