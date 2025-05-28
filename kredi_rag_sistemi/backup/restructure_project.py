#!/usr/bin/env python3
"""
Project Restructure Script
Automatically reorganizes the RAG project into a clean structure
"""

import os
import shutil
import glob
from pathlib import Path

def create_new_structure():
    """Create the new directory structure"""
    
    print("🏗️  Yeni proje yapısı oluşturuluyor...")
    
    # New directory structure
    new_dirs = [
        "src/core",
        "src/models", 
        "src/services",
        "src/utils",
        "interfaces/web",
        "interfaces/api",
        "tests/unit",
        "tests/integration", 
        "tests/performance",
        "tests/debug",
        "scripts",
        "configs",
        "docs",
        "docker"
    ]
    
    for dir_path in new_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        # Create __init__.py files for Python packages
        if not dir_path.startswith(('tests', 'scripts', 'configs', 'docs', 'docker')):
            (Path(dir_path) / "__init__.py").touch()
    
    print("✅ Yeni dizin yapısı oluşturuldu")

def move_test_files():
    """Move all test and debug files to tests directory"""
    
    print("🧪 Test dosyaları taşınıyor...")
    
    # Test files to move
    test_files = [
        "test_*.py",
        "debug_*.py", 
        "check_*.py",
        "inspect_*.py"
    ]
    
    moved_count = 0
    for pattern in test_files:
        for file_path in glob.glob(pattern):
            dest = f"tests/debug/{file_path}"
            shutil.move(file_path, dest)
            moved_count += 1
            print(f"  📁 {file_path} -> {dest}")
    
    print(f"✅ {moved_count} test/debug dosyası taşındı")

def consolidate_core_files():
    """Move core application files to src/core"""
    
    print("🔧 Core dosyalar taşınıyor...")
    
    # Move main application files
    core_mappings = {
        "app/core/rag.py": "src/core/rag_engine.py",
        "app/core/pdf_processor.py": "src/core/document_processor.py"
    }
    
    for src, dest in core_mappings.items():
        if os.path.exists(src):
            shutil.move(src, dest)
            print(f"  📁 {src} -> {dest}")
    
    print("✅ Core dosyalar taşındı")

def move_models():
    """Move model files to src/models"""
    
    print("🤖 Model dosyaları taşınıyor...")
    
    model_mappings = {
        "models/llm.py": "src/models/llm_interface.py",
        "models/embeddings.py": "src/models/embeddings.py",
        "models/vector_store.py": "src/models/vector_store.py"
    }
    
    for src, dest in model_mappings.items():
        if os.path.exists(src):
            shutil.move(src, dest)
            print(f"  📁 {src} -> {dest}")
    
    print("✅ Model dosyalar taşındı")

def move_interfaces():
    """Move UI files to interfaces"""
    
    print("🖥️  Interface dosyaları taşınıyor...")
    
    interface_mappings = {
        "app/streamlit_app.py": "interfaces/web/streamlit_app.py",
        "run_streamlit.py": "interfaces/web/run_streamlit.py",
        "run_api.py": "interfaces/api/run_api.py"
    }
    
    for src, dest in interface_mappings.items():
        if os.path.exists(src):
            shutil.move(src, dest)
            print(f"  📁 {src} -> {dest}")
    
    print("✅ Interface dosyalar taşındı")

def move_scripts():
    """Move utility scripts to scripts directory"""
    
    print("📋 Script dosyalar taşınıyor...")
    
    script_files = [
        "clear_vector_db.py",
        "optimize_pdf_process.py",
        "test_optimization.py"
    ]
    
    for script in script_files:
        if os.path.exists(script):
            dest = f"scripts/{script}"
            shutil.move(script, dest)
            print(f"  📁 {script} -> {dest}")
    
    print("✅ Script dosyalar taşındı")

def move_utils():
    """Move utils directory to src/utils"""
    
    print("🔧 Utils dosyaları taşınıyor...")
    
    if os.path.exists("utils"):
        if os.path.exists("src/utils"):
            shutil.rmtree("src/utils")
        shutil.move("utils", "src/utils")
        print("  📁 utils/ -> src/utils/")
    
    print("✅ Utils dosyalar taşındı")

def move_docker_files():
    """Move Docker files to docker directory"""
    
    print("🐳 Docker dosyaları taşınıyor...")
    
    docker_files = [
        "Dockerfile",
        "docker-compose.yml",
        ".dockerignore"
    ]
    
    for docker_file in docker_files:
        if os.path.exists(docker_file):
            dest = f"docker/{docker_file}"
            shutil.move(docker_file, dest)
            print(f"  📁 {docker_file} -> {dest}")
    
    print("✅ Docker dosyalar taşındı")

def move_docs():
    """Move documentation to docs directory"""
    
    print("📚 Dokümantasyon taşınıyor...")
    
    doc_files = [
        "OPTIMIZATION_GUIDE.md",
        "project_restructure_plan.md",
        "README.md"
    ]
    
    for doc_file in doc_files:
        if os.path.exists(doc_file):
            dest = f"docs/{doc_file}"
            shutil.copy2(doc_file, dest)  # Copy instead of move to keep README in root
            if doc_file != "README.md":
                os.remove(doc_file)
            print(f"  📁 {doc_file} -> {dest}")
    
    print("✅ Dokümantasyon taşındı")

def create_config_files():
    """Create configuration files"""
    
    print("⚙️  Konfigürasyon dosyaları oluşturuluyor...")
    
    # Create development config
    dev_config = """# Development Configuration
debug: true
log_level: DEBUG
vector_db_path: ./data/vector_db
model_path: ./models
cache_enabled: true
cache_ttl: 300

# Model settings
llm:
  temperature: 0.1
  max_tokens: 512
  context_length: 2048

# Vector store settings
vector_store:
  top_k: 3
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
"""
    
    with open("configs/development.yaml", "w") as f:
        f.write(dev_config)
    
    # Create .env.example
    env_example = """# Environment Variables Example
PYTHONPATH=./src
DEBUG=true
LOG_LEVEL=INFO

# Model Configuration
MODEL_PATH=./models
VECTOR_DB_PATH=./data/vector_db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Streamlit Configuration
STREAMLIT_PORT=8501
"""
    
    with open(".env.example", "w") as f:
        f.write(env_example)
    
    print("✅ Konfigürasyon dosyaları oluşturuldu")

def update_gitignore():
    """Update .gitignore with new structure"""
    
    print("📝 .gitignore güncelleniyor...")
    
    additional_ignores = """
# New structure specific ignores
/models/*.gguf
/models/*.bin
/data/vector_db/
/data/cache/
/data/temp/
.env
__pycache__/
*.pyc
.pytest_cache/
.coverage
htmlcov/
.DS_Store
"""
    
    with open(".gitignore", "a") as f:
        f.write(additional_ignores)
    
    print("✅ .gitignore güncellendi")

def cleanup_empty_dirs():
    """Remove empty directories"""
    
    print("🧹 Boş dizinler temizleniyor...")
    
    empty_dirs = ["app", "temp_uploads"]
    
    for dir_name in empty_dirs:
        if os.path.exists(dir_name) and not os.listdir(dir_name):
            os.rmdir(dir_name)
            print(f"  🗑️  {dir_name} silindi")
    
    print("✅ Temizlik tamamlandı")

def main():
    """Main restructure function"""
    
    print("🚀 KREDI RAG SISTEMI - PROJE YAPILANDIRMA")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app") or not os.path.exists("models"):
        print("❌ Yanlış dizin! Lütfen kredi_rag_sistemi/ dizininde çalıştırın")
        return
    
    try:
        # Create backup
        print("💾 Backup oluşturuluyor...")
        if not os.path.exists("backup"):
            shutil.copytree(".", "backup", ignore=shutil.ignore_patterns("backup", ".git", "__pycache__", "*.pyc"))
        
        # Execute restructuring steps
        create_new_structure()
        move_test_files() 
        consolidate_core_files()
        move_models()
        move_interfaces()
        move_scripts()
        move_utils()
        move_docker_files()
        move_docs()
        create_config_files()
        update_gitignore()
        cleanup_empty_dirs()
        
        print("\n" + "=" * 50)
        print("✅ PROJE YAPILANDIRMA TAMAMLANDI!")
        print("🎉 Yeni yapı:")
        print("   📁 src/          - Ana kaynak kod")
        print("   📁 interfaces/   - UI ve API")
        print("   📁 tests/        - Tüm testler")
        print("   📁 scripts/      - Utility scriptler")
        print("   📁 configs/      - Konfigürasyon")
        print("   📁 docs/         - Dokümantasyon")
        print("   📁 docker/       - Container dosyaları")
        print("\n💡 Sonraki adım: Import path'lerini güncelle")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {str(e)}")
        print("💾 Backup klasöründen geri yükleyebilirsiniz")

if __name__ == "__main__":
    main() 