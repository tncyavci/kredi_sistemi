#!/usr/bin/env python3
"""
Google Colab Pro Plus için hızlı test ve demo scripti
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def check_environment():
    """Check Colab environment and capabilities"""
    print("🔍 Environment Check")
    print("=" * 50)
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Google Colab environment detected")
        
        # Check GPU availability
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if result.returncode == 0:
                print("🎯 GPU available:")
                # Extract GPU info
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Tesla' in line or 'V100' in line or 'A100' in line or 'T4' in line:
                        print(f"   {line.strip()}")
                        break
                os.environ["COLAB_PRO_PLUS"] = "1"
                print("🚀 Pro Plus features enabled")
            else:
                print("⚠️ No GPU detected - using CPU mode")
                os.environ["COLAB_PRO_PLUS"] = "0"
        except FileNotFoundError:
            print("⚠️ nvidia-smi not found - using CPU mode")
            os.environ["COLAB_PRO_PLUS"] = "0"
            
        # Check memory
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        print(f"💾 Total RAM: {total_gb:.1f}GB")
        
        if total_gb > 25:
            print("🚀 High-RAM runtime detected")
        elif total_gb > 12:
            print("📊 Standard runtime")
        else:
            print("📱 Basic runtime")
            
    except ImportError:
        print("❌ Not running in Google Colab")
        return False
    
    return True

def quick_setup():
    """Quick setup for testing"""
    print("\n🚀 Quick Setup for Testing")
    print("=" * 50)
    
    # Set environment variables
    os.environ["COLAB_ENV"] = "1"
    is_pro_plus = os.getenv("COLAB_PRO_PLUS", "0") == "1"
    
    if is_pro_plus:
        print("🎯 Setting up Pro Plus optimizations...")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["FORCE_CPU"] = "0"
    else:
        print("📱 Setting up free tier optimizations...")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["FORCE_CPU"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Create directories
    directories = ["data/processed", "models/embeddings", "logs", "cache", "test_pdfs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Directories created")

def install_requirements():
    """Install required packages based on environment"""
    print("\n📦 Installing Requirements")
    print("=" * 50)
    
    is_pro_plus = os.getenv("COLAB_PRO_PLUS", "0") == "1"
    
    if is_pro_plus and os.path.exists("requirements_colab_pro.txt"):
        print("🚀 Installing Pro Plus packages...")
        requirements_file = "requirements_colab_pro.txt"
    else:
        print("📱 Installing standard packages...")
        requirements_file = "requirements_colab.txt"
    
    if os.path.exists(requirements_file):
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q",
            "-r", requirements_file
        ], check=True)
        print(f"✅ Installed from {requirements_file}")
    else:
        print("⚠️ Requirements file not found, installing basic packages...")
        basic_packages = [
            "streamlit>=1.28.0",
            "langchain>=0.1.4", 
            "sentence-transformers>=2.2.2",
            "faiss-cpu>=1.7.4",
            "PyPDF2>=3.0.1",
            "pyngrok>=5.1.0",
            "psutil>=5.9.0"
        ]
        
        for package in basic_packages:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"   ✅ {package}")

def test_pdf_processing():
    """Test PDF processing with sample files"""
    print("\n📄 Testing PDF Processing")
    print("=" * 50)
    
    # Check for test PDFs
    test_pdf_dir = Path("test_pdfs")
    pdf_files = list(test_pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("⚠️ No PDF files found in test_pdfs directory")
        print("💡 Upload some PDF files first:")
        print("""
        from google.colab import files
        uploaded = files.upload()
        
        # Move files to test_pdfs
        import shutil
        for filename in uploaded.keys():
            shutil.move(filename, f"test_pdfs/{filename}")
        """)
        return False
    
    print(f"📊 Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        size_mb = pdf_file.stat().st_size / (1024*1024)
        print(f"   📄 {pdf_file.name} ({size_mb:.1f}MB)")
    
    # Quick processing test
    try:
        sys.path.append('.')
        from app.core.pdf_processor import PDFProcessor
        
        print("🔄 Testing PDF processor...")
        processor = PDFProcessor(str(test_pdf_dir))
        
        # Process first PDF only for testing
        first_pdf = pdf_files[0]
        print(f"🧪 Testing with {first_pdf.name}...")
        
        start_time = time.time()
        text = processor.extract_text_from_pdf(str(first_pdf))
        end_time = time.time()
        
        print(f"✅ Extracted {len(text)} characters in {end_time-start_time:.2f}s")
        print(f"📝 Sample text: {text[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ PDF processing test failed: {e}")
        return False

def test_rag_system():
    """Test RAG system initialization"""
    print("\n🧠 Testing RAG System")
    print("=" * 50)
    
    try:
        sys.path.append('.')
        from app.config_colab import get_config
        
        config = get_config()
        print("✅ Configuration loaded successfully")
        
        # Test embedding model loading
        print("🔄 Testing embedding model...")
        from sentence_transformers import SentenceTransformer
        
        model_name = config["model"]["embedding_model"]
        embedder = SentenceTransformer(model_name)
        
        # Test embedding
        test_text = "Bu bir test metnidir."
        embedding = embedder.encode([test_text])
        print(f"✅ Embedding created: {embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        return False

def start_streamlit():
    """Start Streamlit application with ngrok tunnel"""
    print("\n🌐 Starting Streamlit Application")
    print("=" * 50)
    
    try:
        # Install pyngrok if not available
        try:
            import pyngrok
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "pyngrok"])
            import pyngrok
        
        from pyngrok import ngrok
        
        # Kill existing tunnels
        ngrok.kill()
        time.sleep(2)
        
        # Start Streamlit
        print("🚀 Starting Streamlit...")
        proc = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run",
            "streamlit_colab.py",
            "--server.port=8501",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
            "--server.address=0.0.0.0"
        ])
        
        time.sleep(10)  # Wait for Streamlit to start
        
        # Create tunnel
        print("🌐 Creating ngrok tunnel...")
        public_url = ngrok.connect(8501)
        
        print("=" * 60)
        print("🎉 SUCCESS! Your Kredi RAG System is running!")
        print(f"🌐 Access URL: {public_url}")
        print("📱 Click the link to access your application!")
        print("=" * 60)
        
        return proc, public_url
        
    except Exception as e:
        print(f"❌ Failed to start Streamlit: {e}")
        return None, None

def performance_benchmark():
    """Run performance benchmark"""
    print("\n⚡ Performance Benchmark")
    print("=" * 50)
    
    import time
    import psutil
    
    is_pro_plus = os.getenv("COLAB_PRO_PLUS", "0") == "1"
    
    # Memory benchmark
    memory = psutil.virtual_memory()
    print(f"💾 Available RAM: {memory.available / (1024**3):.1f}GB")
    print(f"💾 RAM Usage: {memory.percent:.1f}%")
    
    # CPU benchmark
    cpu_count = psutil.cpu_count()
    print(f"🔧 CPU Cores: {cpu_count}")
    
    # Quick computation benchmark
    print("🧮 Running computation benchmark...")
    start_time = time.time()
    
    # Simple matrix operations
    import numpy as np
    size = 1000 if is_pro_plus else 500
    a = np.random.random((size, size))
    b = np.random.random((size, size))
    c = np.dot(a, b)
    
    end_time = time.time()
    compute_time = end_time - start_time
    
    print(f"⏱️ Matrix multiplication ({size}x{size}): {compute_time:.2f}s")
    
    if is_pro_plus:
        # GPU benchmark if available
        try:
            import torch
            if torch.cuda.is_available():
                print("🎯 GPU benchmark...")
                device = torch.device("cuda")
                a_gpu = torch.randn(1000, 1000).to(device)
                b_gpu = torch.randn(1000, 1000).to(device)
                
                start_time = time.time()
                c_gpu = torch.matmul(a_gpu, b_gpu)
                torch.cuda.synchronize()
                end_time = time.time()
                
                gpu_time = end_time - start_time
                print(f"⚡ GPU matrix multiplication: {gpu_time:.2f}s")
                print(f"🚀 GPU speedup: {compute_time/gpu_time:.1f}x")
        except:
            print("⚠️ GPU benchmark failed")

def main():
    """Main function to run all tests"""
    print("🏦 Kredi RAG System - Colab Pro Plus Quick Test")
    print("=" * 60)
    
    if not check_environment():
        return
    
    quick_setup()
    
    print("\n🧪 Running Tests...")
    
    # Install requirements
    install_requirements()
    
    # Performance benchmark
    performance_benchmark()
    
    # Test PDF processing
    test_pdf_processing()
    
    # Test RAG system
    test_rag_system()
    
    print("\n🎯 All tests completed!")
    print("\n🚀 Ready to start the application!")
    print("Run: !python run_colab.py")

if __name__ == "__main__":
    main() 