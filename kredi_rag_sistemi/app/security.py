import secrets
import os
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from dotenv import load_dotenv
from pathlib import Path

# .env dosyasını yükle (varsa)
env_path = Path(__file__).parent.parent / '.env'
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)

# Güvenlik yapılandırmaları
API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "changeme")
API_KEY = os.getenv("API_KEY", secrets.token_urlsafe(32))

# HTTP Temel Kimlik Doğrulama
security = HTTPBasic()

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    """
    HTTP Temel Kimlik Doğrulama ile kullanıcı doğrulama
    
    Args:
        credentials: HTTP Temel Kimlik Bilgileri
        
    Returns:
        Doğrulanmış kullanıcı adı
        
    Raises:
        HTTPException: Kimlik doğrulama başarısız olursa
    """
    correct_username = secrets.compare_digest(credentials.username, API_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, API_PASSWORD)
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Geçersiz kimlik bilgileri",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

def verify_api_key(api_key: str):
    """
    API anahtarı doğrulama
    
    Args:
        api_key: Kontrol edilecek API anahtarı
        
    Returns:
        bool: API anahtarı geçerliyse True
    """
    return secrets.compare_digest(api_key, API_KEY)

def generate_secure_token(length=32):
    """
    Güvenli bir rasgele token oluşturur
    
    Args:
        length: Token uzunluğu
        
    Returns:
        str: Güvenli rasgele token
    """
    return secrets.token_urlsafe(length)

def get_env_file_template():
    """
    Örnek .env dosyası şablonu oluşturur
    
    Returns:
        str: .env dosyası içeriği
    """
    return f"""# Güvenlik Ayarları
API_USERNAME=admin
API_PASSWORD={secrets.token_urlsafe(16)}
API_KEY={secrets.token_urlsafe(32)}

# PDF İşleme Ayarları
MAX_PDF_SIZE=10485760  # 10 MB
ALLOWED_EXTENSIONS=pdf,PDF

# Veritabanı Ayarları
VECTOR_DB_TYPE=chroma
ENCRYPTION_ENABLED=false
# ENCRYPTION_KEY={secrets.token_hex(16)}  # Şifreleme için kullanılacak anahtar
"""

# .env dosyası oluşturma yardımcısı
def create_env_file(force=False):
    """
    .env dosyası oluşturur
    
    Args:
        force: True ise, mevcut dosyanın üzerine yazar
        
    Returns:
        bool: Başarılıysa True
    """
    env_path = Path(__file__).parent.parent / '.env'
    
    if os.path.exists(env_path) and not force:
        return False
    
    with open(env_path, 'w') as f:
        f.write(get_env_file_template())
    
    return True

if __name__ == "__main__":
    # Bu modül doğrudan çalıştırıldığında .env dosyası oluştur
    if create_env_file():
        print(f".env dosyası oluşturuldu: {env_path}")
    else:
        print(f".env dosyası zaten var: {env_path}")
        print("Üzerine yazmak için: python -c 'from app.security import create_env_file; create_env_file(force=True)'")
    
    # Güvenlik bilgilerini yazdır
    print("\nGüvenlik Bilgileri:")
    print(f"API_USERNAME: {API_USERNAME}")
    print(f"API_PASSWORD: {API_PASSWORD}")
    print(f"API_KEY: {API_KEY}") 