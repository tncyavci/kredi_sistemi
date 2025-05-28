#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tablo belgelerini özellikle kontrol eden script
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from app.core.rag import KrediRAG

def check_table_documents():
    """Tablo belgelerini özellikle kontrol et"""
    print("🔍 Tablo belgelerini özellikle arıyorum...")
    
    try:
        # RAG sistemi başlat
        model_path = "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        vector_db_path = "./data/vector_db"
        rag = KrediRAG(model_path=model_path, vector_db_path=vector_db_path)
        
        # Tüm belgeleri al
        all_results = rag.vector_store.collection.get()
        
        print(f"📊 Toplam belge sayısı: {len(all_results['ids'])}")
        
        # Type'lara göre grupla
        type_counts = {}
        table_documents = []
        
        for i, metadata in enumerate(all_results['metadatas']):
            doc_type = metadata.get('type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
            if doc_type == 'table':
                table_documents.append({
                    'id': all_results['ids'][i],
                    'metadata': metadata,
                    'text': all_results['documents'][i] if all_results['documents'] else ''
                })
        
        print("\n📋 Belge türleri:")
        for doc_type, count in type_counts.items():
            print(f"  - {doc_type}: {count} adet")
        
        print(f"\n🎯 Bulunan tablo belgesi sayısı: {len(table_documents)}")
        
        if table_documents:
            print("\n📄 Tablo belgeleri:")
            for i, doc in enumerate(table_documents):
                print(f"\n--- TABLO BELGESİ {i+1} ---")
                print(f"ID: {doc['id']}")
                print(f"Metadata: {doc['metadata']}")
                print(f"Content length: {len(doc['text'])} karakters")
                print(f"Content preview: {doc['text'][:200]}...")
        else:
            print("❌ Hiç tablo belgesi bulunamadı!")
            
            # En yakın sonuçları ara
            print("\n🔍 'table' kelimesini arıyorum...")
            search_results = rag.vector_store.query_documents(
                query_text="table tablo",
                top_k=3
            )
            
            for i, result in enumerate(search_results):
                metadata = result.get('metadata', {})
                print(f"\nSonuç {i+1}:")
                print(f"  ID: {result.get('id')}")
                print(f"  Type: {metadata.get('type')}")
                print(f"  Distance: {result.get('distance')}")
                
        # ChromaDB collection info
        print(f"\n📈 ChromaDB Collection Info:")
        try:
            collection_count = rag.vector_store.collection.count()
            print(f"  Collection count: {collection_count}")
        except Exception as e:
            print(f"  Error getting collection count: {e}")
        
    except Exception as e:
        print(f"❌ Hata: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_table_documents() 