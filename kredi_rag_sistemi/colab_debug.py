#!/usr/bin/env python3
"""
Google Colab sorun teşhis ve debug scripti
"""

import os
import sys
import subprocess
import traceback
from pathlib import Path

def print_separator(title):
    """Print a nice separator"""
    print("\n" + "="*60)
    print(f"🔍 {title}")
    print("="*60)

def check_basic_environment():
    """Check basic Python environment"""
    print_separator("BASIC ENVIRONMENT CHECK")
    
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.executable}")
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Google Colab detected")
        
        # Check Colab version
        try:
            from google.colab import __version__
            print(f"Colab version: {__version__}")
        except:
            print("Colab version: Unknown")
            
    except ImportError:
        print("❌ Not running in Google Colab")
        return False
    
    return True

def check_system_resources():
    """Check system resources"""
    print_separator("SYSTEM RESOURCES")
    
    try:
        import psutil
        
        # Memory info
        memory = psutil.virtual_memory()
        print(f"💾 Total RAM: {memory.total / (1024**3):.1f}GB")
        print(f"💾 Available RAM: {memory.available / (1024**3):.1f}GB")
        print(f"💾 Used RAM: {memory.percent:.1f}%")
        
        # CPU info
        print(f"🔧 CPU cores: {psutil.cpu_count()}")
        print(f"🔧 CPU usage: {psutil.cpu_percent()}%")
        
        # Disk info
        disk = psutil.disk_usage('/')
        print(f"💽 Total disk: {disk.total / (1024**3):.1f}GB")
        print(f"💽 Free disk: {disk.free / (1024**3):.1f}GB")
        
    except ImportError:
        print("⚠️ psutil not available, installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil
        memory = psutil.virtual_memory()
        print(f"💾 Total RAM: {memory.total / (1024**3):.1f}GB")

def check_gpu():
    """Check GPU availability"""
    print_separator("GPU CHECK")
    
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("🎯 GPU available:")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Tesla' in line or 'V100' in line or 'A100' in line or 'T4' in line:
                    print(f"   {line.strip()}")
            
            # Check CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"🔥 CUDA available: {torch.cuda.get_device_name()}")
                    print(f"🔥 CUDA version: {torch.version.cuda}")
                else:
                    print("⚠️ PyTorch installed but CUDA not available")
            except ImportError:
                print("⚠️ PyTorch not installed")
                
        else:
            print("❌ No GPU detected")
            
    except FileNotFoundError:
        print("❌ nvidia-smi not found")

def check_project_structure():
    """Check project file structure"""
    print_separator("PROJECT STRUCTURE")
    
    current_dir = Path(".")
    required_files = [
        "requirements_colab.txt",
        "requirements_colab_pro.txt", 
        "colab_setup.py",
        "run_colab.py",
        "streamlit_colab.py",
        "app/config_colab.py",
        "app/core/pdf_processor.py"
    ]
    
    print("📁 Checking required files:")
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING!")
    
    # Check directories
    required_dirs = ["app", "app/core", "utils", "models", "test_pdfs"]
    print("\n📁 Checking directories:")
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ❌ {dir_path}/ - MISSING!")
            
    # List test PDFs
    test_pdfs = list(Path("test_pdfs").glob("*.pdf"))
    print(f"\n📄 Test PDFs found: {len(test_pdfs)}")
    for pdf in test_pdfs:
        size_mb = pdf.stat().st_size / (1024*1024)
        print(f"   📄 {pdf.name} ({size_mb:.1f}MB)")

def test_basic_imports():
    """Test basic package imports"""
    print_separator("BASIC IMPORTS TEST")
    
    basic_packages = [
        "os", "sys", "subprocess", "pathlib",
        "json", "time", "logging"
    ]
    
    for package in basic_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")

def test_required_packages():
    """Test required package imports"""
    print_separator("REQUIRED PACKAGES TEST")
    
    required_packages = [
        ("streamlit", "streamlit"),
        ("pyngrok", "pyngrok"), 
        ("psutil", "psutil"),
        ("pandas", "pandas"),
        ("numpy", "numpy")
    ]
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError as e:
            print(f"❌ {package_name}: {e}")
            print(f"   💡 Install with: !pip install {package_name}")

def test_ai_packages():
    """Test AI/ML package imports"""
    print_separator("AI/ML PACKAGES TEST")
    
    ai_packages = [
        ("sentence-transformers", "sentence_transformers"),
        ("langchain", "langchain"),
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("faiss", "faiss")
    ]
    
    for package_name, import_name in ai_packages:
        try:
            module = __import__(import_name)
            if hasattr(module, '__version__'):
                version = module.__version__
                print(f"✅ {package_name} v{version}")
            else:
                print(f"✅ {package_name}")
        except ImportError as e:
            print(f"❌ {package_name}: {e}")
            print(f"   💡 Install with: !pip install {package_name}")

def test_pdf_packages():
    """Test PDF processing packages"""
    print_separator("PDF PACKAGES TEST")
    
    pdf_packages = [
        ("PyPDF2", "PyPDF2"),
        ("pdfplumber", "pdfplumber"),
        ("PyMuPDF", "fitz"),
        ("pytesseract", "pytesseract"),
        ("pdf2image", "pdf2image")
    ]
    
    for package_name, import_name in pdf_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError as e:
            print(f"❌ {package_name}: {e}")
            print(f"   💡 Install with: !pip install {package_name}")

def test_project_imports():
    """Test project-specific imports"""
    print_separator("PROJECT IMPORTS TEST")
    
    # Add current directory to path
    if "." not in sys.path:
        sys.path.append(".")
    
    project_modules = [
        ("app.config_colab", "get_config"),
        ("app.core.pdf_processor", "PDFProcessor")
    ]
    
    for module_name, class_name in project_modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
        except AttributeError as e:
            print(f"⚠️ {module_name}: Class {class_name} not found")
        except Exception as e:
            print(f"❌ {module_name}: {e}")

def run_minimal_test():
    """Run minimal functionality test"""
    print_separator("MINIMAL FUNCTIONALITY TEST")
    
    try:
        # Test environment setup
        os.environ["COLAB_ENV"] = "1"
        os.environ["COLAB_PRO_PLUS"] = "1"  # Assume Pro Plus
        print("✅ Environment variables set")
        
        # Test config loading
        sys.path.append(".")
        from app.config_colab import get_config
        config = get_config()
        print("✅ Configuration loaded")
        
        # Test basic embedding
        from sentence_transformers import SentenceTransformer
        model_name = config["model"]["embedding_model"]
        print(f"🔄 Loading model: {model_name}")
        
        embedder = SentenceTransformer(model_name)
        test_text = "Test sentence"
        embedding = embedder.encode([test_text])
        print(f"✅ Embedding test successful: {embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Minimal test failed: {e}")
        traceback.print_exc()
        return False

def provide_solutions():
    """Provide solution recommendations"""
    print_separator("SOLUTION RECOMMENDATIONS")
    
    print("🔧 COMMON SOLUTIONS:")
    print()
    print("1. 📦 PACKAGE INSTALLATION:")
    print("   !pip install -r requirements_colab_pro.txt")
    print("   # or for free tier:")
    print("   !pip install -r requirements_colab.txt")
    print()
    
    print("2. 🔄 RUNTIME RESTART:")
    print("   - Go to Runtime > Restart runtime")
    print("   - Then run setup again")
    print()
    
    print("3. 📁 DIRECTORY SETUP:")
    print("   !python colab_setup.py")
    print()
    
    print("4. 🎯 GPU RUNTIME (Pro Plus):")
    print("   - Runtime > Change runtime type")
    print("   - Hardware accelerator: GPU")
    print("   - Runtime shape: High-RAM")
    print()
    
    print("5. ⚡ QUICK FIX COMMANDS:")
    print("   # Install basic requirements")
    print("   !pip install streamlit pyngrok psutil")
    print("   !pip install sentence-transformers langchain")
    print("   !pip install PyPDF2 pdfplumber")
    print()
    
    print("6. 🧪 STEP BY STEP DEBUG:")
    print("   !python colab_debug.py")
    print("   !python quick_test_colab.py")
    print("   !python run_colab.py")

def main():
    """Main debug function"""
    print("🏦 Kredi RAG System - Colab Debug Tool")
    print("🔍 Diagnosing your Colab environment...")
    
    # Run all checks
    if not check_basic_environment():
        print("❌ Not running in Colab environment!")
        return
    
    check_system_resources()
    check_gpu() 
    check_project_structure()
    test_basic_imports()
    test_required_packages()
    test_ai_packages()
    test_pdf_packages()
    test_project_imports()
    
    # Run minimal test
    success = run_minimal_test()
    
    # Provide solutions
    provide_solutions()
    
    if success:
        print_separator("SUCCESS")
        print("🎉 Basic functionality works!")
        print("🚀 Try running: !python run_colab.py")
    else:
        print_separator("ISSUES DETECTED")
        print("❌ Some issues found. Please follow the solutions above.")

if __name__ == "__main__":
    main() 