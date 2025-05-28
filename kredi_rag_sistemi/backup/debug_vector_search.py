#!/usr/bin/env python3
"""
Debug vector search for table documents
"""

from models.vector_store import SecureVectorStore
import json

def debug_vector_search():
    print("🔍 Vector search debug başlıyor...")
    
    vector_store = SecureVectorStore(
        persist_directory='./data/vector_db',
        collection_name='kredi_rag_documents'
    )
    
    count = vector_store.get_document_count()
    print(f"📊 Toplam belge sayısı: {count}")
    
    # ChromaDB collection'ını direkt sorgula
    if hasattr(vector_store, 'collection'):
        collection = vector_store.collection
        
        # Tüm belgeleri çek (metadata ile)
        print("📚 Tüm belgeler alınıyor...")
        
        try:
            # Küçük batch'ler halinde al
            all_data = collection.get(limit=10)  # İlk 10 belgeyi al
            
            docs = all_data.get('documents', [])
            metadatas = all_data.get('metadatas', [])
            ids = all_data.get('ids', [])
            
            print(f"📊 Alınan belge sayısı: {len(docs)}")
            
            # Metadata analizi
            doc_types = {}
            for i, metadata in enumerate(metadatas):
                doc_type = metadata.get('type', 'unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                
                # İlk birkaç belgeyi detaylı göster
                if i < 5:
                    print(f"\n📄 Belge {i+1}:")
                    print(f"ID: {ids[i]}")
                    print(f"Type: {doc_type}")
                    print(f"Metadata keys: {list(metadata.keys())}")
                    if doc_type == 'table':
                        print(f"🎯 TABLO BELGESİ BULUNDU!")
                        print(f"Content preview: {docs[i][:200]}...")
                    print("-" * 50)
            
            print(f"\n📋 Belge türleri (ilk 10 belge):")
            for doc_type, count in doc_types.items():
                print(f"  - {doc_type}: {count} adet")
            
            # Tablo belgelerini özel olarak ara
            table_search_results = collection.get(
                where={"type": "table"},
                limit=10
            )
            
            table_docs = table_search_results.get('documents', [])
            table_metadatas = table_search_results.get('metadatas', [])
            table_ids = table_search_results.get('ids', [])
            
            print(f"\n🎯 Metadata filtresi ile bulunan tablo sayısı: {len(table_docs)}")
            
            if table_docs:
                print("✅ Tablo belgeleri metadata filtresi ile bulundu!")
                for i, (doc, metadata, doc_id) in enumerate(zip(table_docs, table_metadatas, table_ids)):
                    print(f"\n📊 TABLO {i+1}:")
                    print(f"ID: {doc_id}")
                    print(f"Metadata: {json.dumps(metadata, indent=2, ensure_ascii=False)}")
                    print(f"Content ({len(doc)} karakter): {doc[:300]}...")
            else:
                print("❌ Metadata filtresi ile hiç tablo belgesi bulunamadı!")
                
                # Tüm metadata değerlerini kontrol et
                print("\n🔍 Tüm metadata type değerleri:")
                all_data_full = collection.get()
                all_metadatas = all_data_full.get('metadatas', [])
                
                unique_types = set()
                for metadata in all_metadatas:
                    unique_types.add(metadata.get('type', 'None'))
                
                print(f"Unique types: {sorted(unique_types)}")
                
        except Exception as e:
            print(f"❌ ChromaDB sorgu hatası: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Semantic search dene
    print(f"\n{'='*60}")
    print("SEMANTIC SEARCH TESTLERİ:")
    
    search_queries = [
        "tablo",
        "table", 
        "finansal durum tablosu",
        "bilanço",
        "varlık",
        "toplam varlıklar",
        "2022",
        "TABLO 1",
        "FİNANSAL DURUM TABLOSU"
    ]
    
    for query in search_queries:
        print(f"\n🔍 Sorgu: '{query}'")
        results = vector_store.query_documents(query, top_k=5)
        print(f"Sonuç sayısı: {len(results)}")
        
        table_count = 0
        for i, doc in enumerate(results):
            doc_type = doc.get('metadata', {}).get('type', 'unknown')
            if doc_type == 'table':
                table_count += 1
                print(f"  🎯 [{i+1}] TABLO BELGESİ - ID: {doc.get('id', 'unknown')}")
            else:
                print(f"  📄 [{i+1}] {doc_type} - ID: {doc.get('id', 'unknown')}")
        
        print(f"Tablo belge sayısı: {table_count}")

if __name__ == "__main__":
    debug_vector_search() 