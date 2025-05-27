#!/usr/bin/env python3
"""
Google Colab için Streamlit launcher
"""

import os
import subprocess
import time
import signal
import sys
from threading import Thread

def setup_environment():
    """Setup environment variables for Colab"""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["FORCE_CPU"] = "1"
    os.environ["COLAB_ENV"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    print("✅ Environment variables set for Colab")

def install_ngrok():
    """Install and setup ngrok if not available"""
    try:
        import pyngrok
        print("✅ pyngrok is already installed")
    except ImportError:
        print("📦 Installing pyngrok...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyngrok"], check=True)
        print("✅ pyngrok installed successfully")

def run_streamlit_with_tunnel():
    """Run Streamlit with ngrok tunnel for Colab"""
    
    # Setup environment
    setup_environment()
    
    # Install ngrok if needed
    install_ngrok()
    
    from pyngrok import ngrok
    
    # Set ngrok auth token if available
    ngrok_token = os.getenv("NGROK_AUTH_TOKEN")
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)
        print("✅ Ngrok auth token set")
    else:
        print("⚠️ No ngrok auth token found. You may hit connection limits.")
        print("💡 Get a free token from: https://ngrok.com/")
    
    try:
        # Kill any existing ngrok tunnels
        ngrok.kill()
        time.sleep(2)
        
        print("🚀 Starting Streamlit app...")
        
        # Start streamlit in background
        proc = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_colab.py",
            "--server.port=8501",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
            "--server.address=0.0.0.0"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for streamlit to start
        print("⏳ Waiting for Streamlit to start...")
        time.sleep(10)
        
        # Create tunnel
        print("🌐 Creating ngrok tunnel...")
        public_url = ngrok.connect(8501)
        
        print("=" * 60)
        print("🎉 SUCCESS! Your RAG system is now running!")
        print(f"🌐 Access your app at: {public_url}")
        print("📱 Click the link above to access your Kredi RAG system!")
        print("=" * 60)
        print("\n💡 Tips:")
        print("- Upload PDF files using the sidebar")
        print("- Ask questions in Turkish")
        print("- Monitor memory usage in the sidebar")
        print("- Use 'Bellek Temizle' if you run out of memory")
        print("\n⚠️ Keep this cell running to maintain the tunnel!")
        
        # Keep the process running
        try:
            while True:
                # Check if streamlit process is still running
                if proc.poll() is not None:
                    print("❌ Streamlit process died. Restarting...")
                    break
                
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            
        finally:
            # Cleanup
            print("🧹 Cleaning up...")
            
            # Kill streamlit process
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            
            # Kill ngrok tunnels
            ngrok.kill()
            print("✅ Cleanup completed")
            
    except Exception as e:
        print(f"❌ Error starting application: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you've run colab_setup.py first")
        print("2. Check if all requirements are installed")
        print("3. Try restarting the runtime")
        
        return None

def quick_start():
    """Quick start function for Jupyter cells"""
    print("🚀 Kredi RAG System - Quick Start")
    print("=" * 50)
    
    # Check if setup was run
    if not os.path.exists("requirements_colab.txt"):
        print("❌ Setup not detected!")
        print("📋 Please run setup first:")
        print("!python colab_setup.py")
        return
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Google Colab environment detected")
    except ImportError:
        print("⚠️ Not running in Google Colab")
    
    # Start the application
    run_streamlit_with_tunnel()

if __name__ == "__main__":
    quick_start() 