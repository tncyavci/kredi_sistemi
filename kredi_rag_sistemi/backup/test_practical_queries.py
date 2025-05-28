#!/usr/bin/env python3
"""
Test practical financial queries
"""

from app.core.rag import KrediRAG
import os

def test_practical_queries():
    print("🚀 Pratik finansal sorgu testi başlıyor...")
    
    model_path = "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    
    if not os.path.exists(model_path):
        print(f"❌ Model bulunamadı: {model_path}")
        return
    
    # RAG sistemini başlat
    print("🔧 RAG sistemi başlatılıyor...")
    rag = KrediRAG(
        model_path=model_path,
        vector_db_path="./data/vector_db",
        force_recreate_db=False
    )
    
    print(f"📊 Vektör veritabanında {rag.get_document_count()} belge mevcut")
    
    # Praktik sorular
    practical_queries = [
        "Toplam varlıklar ne kadar?",
        "Şirketin finansal durumu nedir?",
        "Varlık dağılımı nasıl?",
        "Bilanço tablosunu göster",
        "Gelir tablosu verileri nelerdir?",
        "En büyük varlık kalemi hangisi?",
        "Yükümlülükler toplamı ne kadar?",
        "Özsermaye tutarı nedir?",
        "Nakit ve nakit benzerleri ne kadar?",
        "Satış gelirleri ne kadar?"
    ]
    
    print(f"\n{'='*80}")
    print("PRATİK FİNANSAL SORGU TESTLERİ:")
    print('='*80)
    
    for i, query in enumerate(practical_queries, 1):
        print(f"\n🔍 SORU {i}: {query}")
        print("-" * 60)
        
        try:
            # Normal query (finansal optimizasyon olmadan)
            result = rag.query(query, use_financial_optimization=False)
            print(f"📊 Normal RAG:")
            print(f"   Yanıt: {result['answer'][:200]}...")
            print(f"   Kaynak sayısı: {len(result.get('source_documents', []))}")
            
            # Financial optimized query
            result_optimized = rag.query(query, use_financial_optimization=True)
            print(f"💰 Financial RAG:")
            print(f"   Yanıt: {result_optimized['answer'][:200]}...")
            print(f"   Kaynak sayısı: {len(result_optimized.get('source_documents', []))}")
            
            # Source analysis
            sources = result_optimized.get('source_documents', [])
            if sources:
                table_sources = [s for s in sources if s.get('metadata', {}).get('type') == 'table']
                text_sources = [s for s in sources if s.get('metadata', {}).get('type') == 'text_chunk']
                
                print(f"   🎯 Tablo kaynakları: {len(table_sources)}")
                print(f"   📄 Metin kaynakları: {len(text_sources)}")
                
                if table_sources:
                    print(f"   ✅ İlk tablo kaynağı: {table_sources[0].get('id', 'unknown')}")
            
        except Exception as e:
            print(f"   ❌ Hata: {str(e)}")
    
    print(f"\n{'='*80}")
    print("ÖZET SONUÇLAR:")
    print('='*80)
    
    # Summary test - direct table query
    print("\n🎯 Direkt tablo sorgularını test edelim:")
    
    direct_table_queries = [
        "finansal durum tablosu",
        "bilanço",
        "tablo verilerini göster",
        "varlık tablosu"
    ]
    
    for query in direct_table_queries:
        print(f"\n🔍 Direkt sorgu: '{query}'")
        try:
            result = rag.query(query, use_financial_optimization=True)
            sources = result.get('source_documents', [])
            table_count = sum(1 for s in sources if s.get('metadata', {}).get('type') == 'table')
            print(f"   📊 Bulunan tablo sayısı: {table_count}")
            print(f"   💬 Yanıt uzunluğu: {len(result.get('answer', ''))}")
            
            if table_count > 0:
                print(f"   ✅ Tablo bulundu!")
        except Exception as e:
            print(f"   ❌ Hata: {str(e)}")

if __name__ == "__main__":
    test_practical_queries() 