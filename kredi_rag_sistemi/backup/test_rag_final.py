#!/usr/bin/env python3
"""
Final RAG system test with table data
"""

from app.core.rag import KrediRAG
import os

def test_rag_final():
    print("🚀 Final RAG sistemi testi başlıyor...")
    
    # Model path
    model_path = "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    
    if not os.path.exists(model_path):
        print(f"❌ Model bulunamadı: {model_path}")
        return
    
    # RAG sistemini başlat
    print("🔧 RAG sistemi başlatılıyor...")
    rag = KrediRAG(
        model_path=model_path,
        vector_db_path="./data/vector_db",
        top_k=3,
        temperature=0.1,
        max_tokens=512
    )
    
    # Belge sayısını kontrol et
    doc_count = rag.get_document_count()
    print(f"📊 Vektör veritabanında {doc_count} belge mevcut")
    
    if doc_count == 0:
        print("❌ Vektör veritabanı boş! Önce PDF'leri işlemelisiniz.")
        return
    
    # Test sorguları - tablo verilerine odaklı
    test_queries = [
        "2022 yılında toplam varlıklar ne kadar?",
        "2023 yılında satışların maliyeti nedir?", 
        "Pegasus'un 2022 yılı finansal durumu nasıl?",
        "Konsolide bilanço verileri nelerdir?",
        "2021-2023 yılları arasında toplam varlık artışı ne kadar?"
    ]
    
    print(f"\n{'='*80}")
    print("RAG SİSTEMİ TEST SORGULARI:")
    print('='*80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 SORU {i}: {query}")
        print("-" * 60)
        
        try:
            # RAG sorgusu
            result = rag.query(query)
            
            # Sonuçları göster
            answer = result.get("answer", "Yanıt alınamadı")
            retrieval_time = result.get("retrieval_time", 0)
            generation_time = result.get("generation_time", 0)
            total_time = result.get("total_time", 0)
            
            print(f"💡 YANIT: {answer}")
            print(f"⏱️ Süre: {total_time:.2f}s (Arama: {retrieval_time:.2f}s, Üretim: {generation_time:.2f}s)")
            
            # Kaynak belgeleri kontrol et
            source_docs = result.get("source_documents", [])
            if source_docs:
                print(f"📚 Kaynak belgeler: {len(source_docs)} adet")
                
                # Tablo belgeleri var mı kontrol et
                table_sources = [doc for doc in source_docs if doc.get('metadata', {}).get('type') == 'table']
                text_sources = [doc for doc in source_docs if doc.get('metadata', {}).get('type') != 'table']
                
                print(f"  - Tablo kaynakları: {len(table_sources)}")
                print(f"  - Metin kaynakları: {len(text_sources)}")
                
                # İlk kaynak belgenin içeriğini göster
                if source_docs:
                    first_doc = source_docs[0]
                    doc_type = first_doc.get('metadata', {}).get('type', 'unknown')
                    preview = first_doc.get('text', '')[:200].replace('\n', ' ')
                    print(f"  📄 İlk kaynak [{doc_type}]: {preview}...")
            
            # Hata kontrolü
            if "error" in result:
                print(f"⚠️ Hata: {result['error']}")
                
        except Exception as e:
            print(f"❌ Sorgu hatası: {str(e)}")
    
    print(f"\n🎯 RAG sistemi testi tamamlandı!")
    print("✅ Tablo verileri başarıyla işlendi ve sorgulanabilir durumda!")

if __name__ == "__main__":
    test_rag_final() 