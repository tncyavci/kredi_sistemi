from llama_cpp import Llama
import os
import warnings
from typing import Dict, List, Optional, Any

# Warnings'leri sustur
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MistralLLM:
    """
    Mistral 7B modeli kullanan LLM sınıfı - CPU Optimized Version
    Bu sınıf, CPU-only modda çalışan Mistral 7B modeli üzerinden
    kredi uygulamaları için RAG sisteminde kullanılacak LLM işlevlerini sağlar.
    """
    
    def __init__(
        self,
        model_path: str = None,
        n_ctx: int = 2048,          # Reduced context for CPU
        n_gpu_layers: int = 0,      # Force CPU-only
        verbose: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 512,      # Reduced for CPU performance
        n_threads: int = 4,         # Optimal for most CPUs
        n_batch: int = 128,         # Smaller batch for CPU
    ):
        """
        MistralLLM sınıfının başlatıcısı - CPU Optimized
        
        Args:
            model_path: GGUF model dosyasının yolu
            n_ctx: Maksimum bağlam uzunluğu (CPU için azaltıldı)
            n_gpu_layers: GPU katman sayısı (0 = CPU-only)
            verbose: Ayrıntılı loglama etkinleştirilsin mi
            temperature: Yanıt oluşturma sıcaklığı (0-1 arası)
            max_tokens: Yanıtta oluşturulacak maksimum token sayısı
            n_threads: CPU thread sayısı
            n_batch: Batch boyutu (CPU için küçük)
        """
        self.model_path = model_path or os.getenv("MISTRAL_MODEL_PATH")
        if not self.model_path:
            raise ValueError("Model yolu belirtilmeli veya MISTRAL_MODEL_PATH ortam değişkeni ayarlanmalıdır.")
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # CPU-only environment'ı zorla
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["FORCE_CPU"] = "1"
        
        print(f"🔄 Mistral model yükleniyor (CPU-only mode)...")
        print(f"📁 Model: {os.path.basename(self.model_path)}")
        print(f"💾 Context: {n_ctx} tokens")
        print(f"🧵 Threads: {n_threads}")
        print(f"📦 Batch: {n_batch}")
        
        try:
            # Model yüklemesi - CPU optimized
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=n_ctx,
                n_gpu_layers=0,         # Force CPU-only
                verbose=verbose,
                n_threads=n_threads,    # CPU thread count
                n_batch=n_batch,       # Batch size for CPU
                use_mmap=True,         # Memory mapping for efficiency
                use_mlock=False,       # Don't lock memory
                low_vram=True          # Optimize for low memory
            )
            print("✅ Mistral model başarıyla yüklendi (CPU-only)")
            
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            print("💡 Çözüm önerileri:")
            print("- Model dosyasının mevcut olduğunu kontrol edin")
            print("- Yeterli RAM'iniz olduğunu kontrol edin (minimum 6GB)")
            print("- Başka programları kapatmayı deneyin")
            raise
        
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Verilen prompt için metni oluşturur - CPU Optimized
        
        Args:
            prompt: Kullanıcı promptu
            system_prompt: Sistem promptu (isteğe bağlı)
            
        Returns:
            Oluşturulan yanıt metni
        """
        try:
            if system_prompt:
                # Mistral formatında system ve user prompt birleştirme
                full_prompt = f"[INST] {system_prompt}\n\n{prompt} [/INST]"
            else:
                full_prompt = f"[INST] {prompt} [/INST]"
            
            # CPU için optimize edilmiş generation
            response = self.llm.create_completion(
                prompt=full_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                echo=False,
                stop=["[INST]", "</s>", "\n\n"],
                top_p=0.9,              # Nucleus sampling
                repeat_penalty=1.1,     # Prevent repetition
                stream=False            # Don't stream for simplicity
            )
            
            return response["choices"][0]["text"].strip()
            
        except Exception as e:
            print(f"❌ Text generation hatası: {e}")
            return f"Yanıt oluşturulamadı: {str(e)}"
    
    def generate_with_context(self, prompt: str, context: List[str], system_prompt: Optional[str] = None) -> str:
        """
        Verilen bağlam bilgileriyle birlikte yanıt oluşturur (RAG için) - Context window optimized
        
        Args:
            prompt: Kullanıcı promptu
            context: Belge vektör tabanından alınan relevan dökümanların listesi
            system_prompt: Sistem promptu (isteğe bağlı)
            
        Returns:
            Oluşturulan yanıt metni
        """
        # Context'i çok agresif şekilde kısıtla (CPU ve context window için)
        limited_context = []
        has_table_data = False
        
        # Maksimum 2 belge, belge başına 300 karakter
        max_docs = min(2, len(context))
        max_chars_per_doc = 300
        
        for i, doc in enumerate(context[:max_docs]):
            if doc and len(doc.strip()) > 0:
                # Tablo verisi kontrolü
                if any(keyword in doc.upper() for keyword in ["TABLO", "TABLE", "|", "2021", "2022", "2023", "2024"]):
                    has_table_data = True
                    max_chars_per_doc = 400  # Tablo için biraz daha fazla
                
                # Sadece en önemli kısmı al
                truncated_doc = doc.strip()[:max_chars_per_doc]
                
                # Cümle ortasında kesilmeyi önle
                if len(truncated_doc) == max_chars_per_doc and '.' in truncated_doc:
                    last_period = truncated_doc.rfind('.')
                    if last_period > max_chars_per_doc // 2:  # En az yarısı kalmalı
                        truncated_doc = truncated_doc[:last_period + 1]
                
                limited_context.append(truncated_doc)
        
        # Context'i birleştir
        context_text = "\n\n".join(limited_context) if limited_context else "İlgili bilgi bulunamadı."
        
        # Tablo verisi varsa özel sistem promptu
        if has_table_data:
            system_prompt = """Sen Türkçe finansal asistansın. Tablolardaki sayıları DİKKATLE oku. 
Sadece tabloda gördüğün sayıları kullan. Uydurmaca yapma."""
        elif system_prompt is None:
            system_prompt = """Sen Türkçe finansal asistansın. Verilen belgelerdeki bilgileri kullan. 
Belgede bilgi yoksa 'Bu bilgi belgelerde mevcut değil' de."""
        
        # Prompt'u kısalt
        short_prompt = prompt[:150] if len(prompt) > 150 else prompt
        
        # Final prompt - çok kısa ve net
        if has_table_data:
            final_prompt = f"""Sistem: {system_prompt}

Tablo Verileri:
{context_text}

Soru: {short_prompt}

Yanıt (sadece tablodaki sayıları kullan):"""
        else:
            final_prompt = f"""Sistem: {system_prompt}

Belgeler:
{context_text}

Soru: {short_prompt}

Yanıt:"""
        
        # Token sayısını kontrol et
        estimated_tokens = len(final_prompt.split()) * 1.3  # Rough estimation
        
        if estimated_tokens > 1400:  # Daha güvenli limit
            # Daha da kısalt
            context_text = context_text[:300] + "..."
            final_prompt = f"""Belgeler: {context_text}

Soru: {short_prompt}

Yanıt:"""
        
        try:
            # LLM'den yanıt al - çok düşük max_tokens
            response = self.llm(
                final_prompt,
                max_tokens=150,  # Daha da düşük
                temperature=0.05,  # Daha deterministik
                stop=["Soru:", "Belgeler:", "Sistem:", "Tablo"],
                echo=False
            )
            
            if response and 'choices' in response and len(response['choices']) > 0:
                answer = response['choices'][0]['text'].strip()
                
                # Boş veya çok kısa yanıtları kontrol et
                if not answer or len(answer) < 5:
                    return "Bu konuda yeterli bilgi bulamadım."
                
                # İngilizce yanıtları Türkçe'ye çevir (basit kontrol)
                if any(word in answer.lower() for word in ['the', 'and', 'for', 'you', 'would', 'need', 'from']):
                    return "Bu bilgi belgelerde mevcut değil."
                
                # Çok uzun yanıtları kısalt
                if len(answer) > 300:
                    answer = answer[:300] + "..."
                
                return answer
            else:
                return "Yanıt oluşturulamadı."
                
        except Exception as e:
            print(f"❌ Text generation hatası: {e}")
            return f"Yanıt oluşturulamadı: {str(e)}"

# Model indirme ve hazırlık fonksiyonu - CPU Optimized
def download_mistral_model(model_name: str = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF", 
                          file_name: str = "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                          save_dir: str = "./models/") -> str:
    """
    Hugging Face'den Mistral 7B modelini indirir (eğer model zaten mevcut değilse).
    CPU için optimize edilmiş versiyon.
    
    Args:
        model_name: Hugging Face'deki model repository adı
        file_name: İndirilecek GGUF dosyasının adı (Q4_K_M CPU için optimal)
        save_dir: Modelin kaydedileceği dizin
        
    Returns:
        İndirilen model dosyasının tam yolu
    """
    import os
    
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("❌ huggingface_hub paketi bulunamadı!")
        print("💡 Çözüm: pip install huggingface_hub")
        raise
    
    # Dizini oluştur
    os.makedirs(save_dir, exist_ok=True)
    
    # Modelin tam yolunu oluştur
    target_path = os.path.join(save_dir, file_name)
    
    # Model zaten var mı kontrol et
    if os.path.exists(target_path):
        print(f"✅ Model zaten mevcut: {target_path}")
        file_size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"📊 Dosya boyutu: {file_size_mb:.1f} MB")
        return target_path
    
    # Model yoksa indir
    print(f"📥 Model indiriliyor: {model_name}/{file_name}")
    print("⏳ Bu işlem birkaç dakika sürebilir...")
    
    try:
        model_path = hf_hub_download(
            repo_id=model_name,
            filename=file_name,
            local_dir=save_dir,
            local_dir_use_symlinks=False  # Symlink kullanma
        )
        
        print(f"✅ Model başarıyla indirildi: {model_path}")
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"📊 Dosya boyutu: {file_size_mb:.1f} MB")
        return model_path
        
    except Exception as e:
        print(f"❌ Model indirme hatası: {e}")
        print("💡 Çözüm önerileri:")
        print("- İnternet bağlantınızı kontrol edin")
        print("- Disk alanınızı kontrol edin (minimum 5GB)")
        print("- Proxy/firewall ayarlarınızı kontrol edin")
        raise

def get_model_info(model_path: str) -> Dict[str, Any]:
    """
    Model hakkında bilgi döndürür
    
    Args:
        model_path: Model dosyasının yolu
        
    Returns:
        Model bilgileri
    """
    if not os.path.exists(model_path):
        return {"exists": False}
    
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    return {
        "exists": True,
        "path": model_path,
        "filename": os.path.basename(model_path),
        "size_mb": round(file_size_mb, 1),
        "size_gb": round(file_size_mb / 1024, 2)
    }