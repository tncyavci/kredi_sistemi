#!/usr/bin/env python3
"""
Simple Streamlit Launcher for Kredi RAG System
Alternative launcher when run.py has issues
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Start Streamlit web interface"""
    
    # Get current directory (should be kredi_rag_sistemi)
    current_dir = Path.cwd()
    print(f"🚀 Kredi RAG System - Streamlit Launcher")
    print(f"📁 Working directory: {current_dir}")
    
    # Check if we're in the right directory
    if not (current_dir / "src").exists() or not (current_dir / "interfaces").exists():
        print("❌ Wrong directory! Please run from kredi_rag_sistemi/")
        print(f"   Current: {current_dir}")
        print(f"   Expected: .../kredi_rag_sistemi/")
        return 1
    
    # Setup environment
    src_path = current_dir / "src"
    current_path = str(current_dir)
    
    # Set Python path
    if "PYTHONPATH" in os.environ:
        os.environ["PYTHONPATH"] = f"{current_path}{os.pathsep}{src_path}{os.pathsep}{os.environ['PYTHONPATH']}"
    else:
        os.environ["PYTHONPATH"] = f"{current_path}{os.pathsep}{src_path}"
    
    # Set other environment variables
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["FORCE_CPU"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Path to Streamlit app
    streamlit_app = current_dir / "interfaces" / "web" / "streamlit_app.py"
    
    if not streamlit_app.exists():
        print(f"❌ Streamlit app not found: {streamlit_app}")
        return 1
    
    print("🌐 Starting Streamlit...")
    print("📱 URL: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(streamlit_app),
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--browser.gatherUsageStats=false",
            "--server.headless=true"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Streamlit stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 