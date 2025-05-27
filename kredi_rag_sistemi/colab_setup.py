#!/usr/bin/env python3
"""
Google Colab için kredi RAG sistemi kurulum scripti
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
from pathlib import Path

def install_system_dependencies():
    """Install system dependencies required for PDF processing"""
    print("📦 Installing system dependencies...")
    
    # Tesseract OCR installation
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run([
        "apt-get", "install", "-y", "-qq",
        "tesseract-ocr",
        "tesseract-ocr-tur",  # Turkish language support
        "tesseract-ocr-eng",  # English language support
        "libtesseract-dev",
        "poppler-utils",      # For pdf2image
        "default-jre"         # For tabula-py
    ], check=True)
    
    print("✅ System dependencies installed successfully")

def install_python_packages():
    """Install Python packages from requirements"""
    print("🐍 Installing Python packages...")
    
    # Check if Pro Plus is available
    is_pro_plus = os.getenv("COLAB_PRO_PLUS", "0") == "1"
    
    if is_pro_plus:
        print("🚀 Installing Pro Plus optimized packages...")
        requirements_file = "requirements_colab_pro.txt"
        
        # Check for GPU availability
        try:
            subprocess.run(["nvidia-smi"], check=True, capture_output=True)
            print("🎯 GPU detected - installing GPU-optimized packages")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️ No GPU detected - falling back to CPU packages")
            requirements_file = "requirements_colab.txt"
    else:
        print("📦 Installing standard Colab packages...")
        requirements_file = "requirements_colab.txt"
    
    # Install packages
    if os.path.exists(requirements_file):
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q",
            "-r", requirements_file
        ], check=True)
        print(f"✅ Packages from {requirements_file} installed successfully")
    else:
        print(f"⚠️ {requirements_file} not found, installing basic packages...")
        basic_packages = [
            "streamlit", "langchain", "sentence-transformers", 
            "faiss-cpu", "PyPDF2", "pyngrok"
        ]
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q"
        ] + basic_packages, check=True)
        print("✅ Basic packages installed successfully")

def setup_directories():
    """Create necessary directories for the project"""
    print("📁 Setting up directories...")
    
    directories = [
        "data/processed",
        "models/embeddings", 
        "logs",
        "test_pdfs",
        "cache"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    print("✅ Directories created successfully")

def download_sample_pdfs():
    """Download sample PDF files for testing"""
    print("📄 Downloading sample PDF files...")
    
    # You can add sample PDF downloads here if needed
    # For now, we'll just create the directory
    os.makedirs("test_pdfs", exist_ok=True)
    
    print("✅ Sample PDFs directory ready")

def setup_colab_environment():
    """Setup Google Colab specific configurations"""
    print("🔧 Configuring for Google Colab...")
    
    # Check if Pro Plus subscription is available
    try:
        import google.colab
        # Try to detect if we have Pro Plus features
        has_pro_plus = True  # Assume Pro Plus for now
        os.environ["COLAB_PRO_PLUS"] = "1" if has_pro_plus else "0"
        
        if has_pro_plus:
            print("🚀 Google Colab Pro Plus detected!")
            print("📊 Enabling GPU acceleration and high-memory features")
            # Enable GPU acceleration for Pro Plus
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            os.environ["FORCE_CPU"] = "0"  # Allow GPU usage
            os.environ["COLAB_ENV"] = "1"
            # Don't disable CUDA for Pro Plus
            print("🎯 GPU acceleration enabled")
        else:
            print("⚠️ Free tier detected, using CPU-only mode")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            os.environ["FORCE_CPU"] = "1"
            os.environ["COLAB_ENV"] = "1"
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            
    except ImportError:
        print("⚠️ Not running in Google Colab, using standard settings")
        os.environ["COLAB_PRO_PLUS"] = "0"
        os.environ["COLAB_ENV"] = "0"
    
    print("✅ Colab environment configured")

def create_colab_launcher():
    """Create a launcher script for Colab"""
    launcher_content = '''
import os
import subprocess
from pyngrok import ngrok
import streamlit as st

def run_streamlit_with_tunnel():
    """Run Streamlit with ngrok tunnel for Colab"""
    
    # Set ngrok auth token if available
    ngrok_token = os.getenv("NGROK_AUTH_TOKEN")
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)
    
    # Start streamlit in background
    proc = subprocess.Popen([
        "streamlit", "run", 
        "app/streamlit_app.py",
        "--server.port=8501",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ])
    
    # Create tunnel
    public_url = ngrok.connect(8501)
    print(f"🌐 Streamlit app is available at: {public_url}")
    print("📱 Click the link above to access your RAG system!")
    
    return proc, public_url

if __name__ == "__main__":
    proc, url = run_streamlit_with_tunnel()
    print(f"✅ App is running at: {url}")
'''
    
    with open("run_colab.py", "w", encoding="utf-8") as f:
        f.write(launcher_content)
    
    print("✅ Colab launcher created")

def main():
    """Main setup function"""
    print("🚀 Setting up Kredi RAG System for Google Colab...")
    print("=" * 60)
    
    try:
        # Check if we're in Colab
        try:
            import google.colab
            print("✅ Google Colab environment detected")
        except ImportError:
            print("⚠️  Not running in Google Colab, but continuing setup...")
        
        # Setup steps
        install_system_dependencies()
        install_python_packages()
        setup_directories()
        download_sample_pdfs()
        setup_colab_environment()
        create_colab_launcher()
        
        print("=" * 60)
        print("🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Upload your PDF files to the 'test_pdfs' directory")
        print("2. Run: python run_colab.py")
        print("3. Click the ngrok URL to access your RAG system")
        print("\n💡 Tips:")
        print("- Use 'files.upload()' in Colab to upload PDF files")
        print("- Set NGROK_AUTH_TOKEN environment variable for better tunneling")
        print("- GPU acceleration will be used if available in Colab")
        
    except Exception as e:
        print(f"❌ Setup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 