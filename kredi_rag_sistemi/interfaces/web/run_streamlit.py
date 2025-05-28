#!/usr/bin/env python3
"""
Streamlit Application Runner
Runs the Kredi RAG Streamlit web interface
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Setup environment variables and paths"""
    # Get project root directory
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    
    # Add src to Python path
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    
    # Set environment variables
    os.environ["PYTHONPATH"] = f"{project_root}{os.pathsep}{project_root / 'src'}"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["FORCE_CPU"] = "1"
    
    print(f"🚀 Project root: {project_root}")
    print(f"🐍 Python path: {os.environ.get('PYTHONPATH', 'Not set')}")

def run_streamlit():
    """Run the Streamlit application"""
    setup_environment()
    
    # Get the streamlit app path
    app_path = Path(__file__).parent / "streamlit_app.py"
    
    if not app_path.exists():
        print(f"❌ Streamlit app not found: {app_path}")
        return
    
    print("🌐 Starting Streamlit application...")
    print("📱 Access URL: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port=8501",
            "--server.address=0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Streamlit uygulaması durduruldu")
    except Exception as e:
        print(f"❌ Hata: {str(e)}")

if __name__ == "__main__":
    run_streamlit() 