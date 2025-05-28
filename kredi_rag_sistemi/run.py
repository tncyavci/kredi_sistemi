#!/usr/bin/env python3
"""
Kredi RAG Sistemi - Ana Çalıştırma Script'i
Main launcher for the Kredi RAG System
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def setup_environment():
    """Setup project environment"""
    project_root = Path(__file__).parent
    
    # Add project and src to Python path
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    
    # Set environment variables
    os.environ["PYTHONPATH"] = f"{project_root}{os.pathsep}{project_root / 'src'}"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["FORCE_CPU"] = "1"
    
    print(f"🚀 Kredi RAG Sistemi - v2.0")
    print(f"📁 Project root: {project_root}")
    print("=" * 50)

def run_streamlit():
    """Run Streamlit web interface"""
    print("🌐 Streamlit Web Interface başlatılıyor...")
    
    app_path = Path(__file__).parent / "interfaces" / "web" / "streamlit_app.py"
    
    if not app_path.exists():
        print(f"❌ Streamlit app bulunamadı: {app_path}")
        return
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Streamlit durduruldu")
    except Exception as e:
        print(f"❌ Streamlit hatası: {str(e)}")

def run_api():
    """Run FastAPI REST interface"""
    print("🔌 FastAPI REST Interface başlatılıyor...")
    
    api_path = Path(__file__).parent / "interfaces" / "api" / "run_api.py"
    
    if not api_path.exists():
        print(f"❌ API script bulunamadı: {api_path}")
        return
    
    try:
        subprocess.run([sys.executable, str(api_path)])
    except KeyboardInterrupt:
        print("\n🛑 API durduruldu")
    except Exception as e:
        print(f"❌ API hatası: {str(e)}")

def run_tests():
    """Run test suite"""
    print("🧪 Test paketi çalıştırılıyor...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pytest",
            "tests/",
            "-v", "--tb=short"
        ])
    except Exception as e:
        print(f"❌ Test hatası: {str(e)}")

def run_optimization_test():
    """Run optimization performance test"""
    print("⚡ Performance optimizasyon testi...")
    
    test_script = Path(__file__).parent / "scripts" / "test_optimization.py"
    
    if test_script.exists():
        try:
            subprocess.run([sys.executable, str(test_script)])
        except Exception as e:
            print(f"❌ Optimization test hatası: {str(e)}")
    else:
        print(f"❌ Test script bulunamadı: {test_script}")

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="Kredi RAG Sistemi Launcher")
    parser.add_argument(
        "mode",
        choices=["web", "api", "test", "optimize"],
        help="Çalıştırılacak mod"
    )
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Run selected mode
    if args.mode == "web":
        run_streamlit()
    elif args.mode == "api":
        run_api()
    elif args.mode == "test":
        run_tests()
    elif args.mode == "optimize":
        run_optimization_test()

if __name__ == "__main__":
    main() 