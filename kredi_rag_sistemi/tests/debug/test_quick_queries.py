#!/usr/bin/env python3
"""
Quick test for financial queries
"""

from app.core.rag import KrediRAG
import os

def test_quick_queries():
    print("🚀 Hızlı finansal sorgu testi başlıyor...")
    
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
    
    # Hızlı test sorular
    quick_queries = [
        "Toplam varlıklar ne kadar?",
        "Bilanço tablosunu göster",
        "Şirketin varlık dağılımı nasıl?"
    ]
    
    print(f"\n{'='*80}")
    print("HIZLI FİNANSAL SORGU TESTLERİ:")
    print('='*80)
    
    for i, query in enumerate(quick_queries, 1):
        print(f"\n🔍 SORU {i}: {query}")
        print("-" * 60)
        
        try:
            # Financial optimized query
            result = rag.query(query, use_financial_optimization=True)
            
            print(f"💰 Financial RAG yanıtı:")
            print(f"   {result['answer'][:300]}...")
            print(f"   Kaynak sayısı: {len(result.get('source_documents', []))}")
            
            # Source analysis
            sources = result.get('source_documents', [])
            if sources:
                table_sources = [s for s in sources if s.get('metadata', {}).get('type') == 'table']
                text_sources = [s for s in sources if s.get('metadata', {}).get('type') == 'text_chunk']
                
                print(f"   🎯 Tablo kaynakları: {len(table_sources)}")
                print(f"   📄 Metin kaynakları: {len(text_sources)}")
                
                if table_sources:
                    print(f"   ✅ İlk tablo kaynağı: {table_sources[0].get('id', 'unknown')}")
                    
                    # Tablo içeriği önizlemesi
                    table_content = table_sources[0].get('text', '')
                    print(f"   📊 Tablo içeriği: {table_content[:150]}...")
            
            print(f"   ⏱️ Süre: {result.get('total_time', 0):.2f}s")
            
        except Exception as e:
            print(f"   ❌ Hata: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_quick_queries() 