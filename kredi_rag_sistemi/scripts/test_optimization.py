#!/usr/bin/env python3
"""
Optimization Test Script
Tests the performance improvements made to the RAG system
"""

import os
import sys
import time
import logging

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "src"))

# Configure CPU-only mode
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["FORCE_CPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.core.rag_engine import KrediRAG
from src.models.llm_interface import download_mistral_model

def test_rag_performance():
    """Test RAG system performance with optimizations"""
    
    print("🚀 RAG Performans Optimizasyon Testi")
    print("=" * 50)
    
    # Setup
    model_dir = os.path.join(ROOT_DIR, "models")
    vector_db_path = os.path.join(ROOT_DIR, "data", "vector_db")
    
    try:
        # Initialize RAG system
        print("📥 Model yükleniyor...")
        model_path = download_mistral_model(save_dir=model_dir)
        
        print("🔧 RAG sistemi başlatılıyor...")
        start_time = time.time()
        
        rag = KrediRAG(
            model_path=model_path,
            vector_db_path=vector_db_path,
            top_k=3
        )
        
        init_time = time.time() - start_time
        print(f"✅ RAG sistemi başlatıldı ({init_time:.2f}s)")
        
        # Check document count
        doc_count = rag.get_document_count()
        print(f"📊 Vektör veritabanında {doc_count} belge mevcut")
        
        if doc_count == 0:
            print("⚠️  Vektör veritabanında belge yok. Önce PDF'leri işleyin.")
            return
        
        # Test queries
        test_queries = [
            "Pegasus'un toplam aktifleri nedir?",
            "2021 yılı gelir tablosu",
            "Özsermaye miktarı",
            "Pegasus'un toplam aktifleri nedir?",  # Duplicate to test cache
            "Bilanço bilgileri"
        ]
        
        print("\n🔍 Test sorguları çalıştırılıyor...")
        
        total_retrieval_time = 0
        total_generation_time = 0
        cache_hits = 0
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Sorgu: '{query}'")
            
            start_time = time.time()
            result = rag.query(query, top_k=3)
            query_time = time.time() - start_time
            
            if "error" not in result:
                retrieval_time = result.get('retrieval_time', 0)
                generation_time = result.get('generation_time', 0)
                found_docs = len(result.get('source_documents', []))
                
                total_retrieval_time += retrieval_time
                total_generation_time += generation_time
                
                # Check if served from cache (very fast query time indicates cache hit)
                if query_time < 1.0 and retrieval_time < 0.1:
                    cache_hits += 1
                    print("   💾 Cache'den servis edildi")
                
                print(f"   ⏱️  Toplam süre: {query_time:.2f}s")
                print(f"   🔍 Arama süresi: {retrieval_time:.2f}s")
                print(f"   🤖 Üretim süresi: {generation_time:.2f}s")
                print(f"   📄 Bulunan belge: {found_docs}")
                print(f"   💬 Yanıt: {result['answer'][:100]}...")
            else:
                print(f"   ❌ Hata: {result.get('error', 'Bilinmeyen hata')}")
        
        # Performance summary
        print("\n" + "=" * 50)
        print("📈 PERFORMANS ÖZETİ")
        print("=" * 50)
        print(f"Toplam sorgu sayısı: {len(test_queries)}")
        print(f"Cache hit sayısı: {cache_hits}")
        print(f"Cache hit oranı: {(cache_hits/len(test_queries)*100):.1f}%")
        print(f"Ortalama arama süresi: {(total_retrieval_time/len(test_queries)):.2f}s")
        print(f"Ortalama üretim süresi: {(total_generation_time/len(test_queries)):.2f}s")
        
        # Test cache functionality
        print("\n🧪 Cache fonksiyonalitesi test ediliyor...")
        
        # Clear cache and test
        rag.clear_query_cache()
        print("   ✅ Cache temizlendi")
        
        # Same query should be slower now
        start_time = time.time()
        result = rag.query(test_queries[0], top_k=3)
        first_time = time.time() - start_time
        
        # Same query again should be faster (from cache)
        start_time = time.time()
        result = rag.query(test_queries[0], top_k=3)
        second_time = time.time() - start_time
        
        print(f"   ⏱️  İlk sorgu: {first_time:.2f}s")
        print(f"   ⏱️  İkinci sorgu (cache): {second_time:.2f}s")
        print(f"   📈 Hız artışı: {(first_time/second_time):.1f}x")
        
        print("\n✅ Optimizasyon testi tamamlandı!")
        
    except Exception as e:
        print(f"❌ Test hatası: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag_performance() 