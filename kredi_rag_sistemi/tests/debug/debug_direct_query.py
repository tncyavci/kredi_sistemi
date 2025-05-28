#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vector store direkt query test
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from models.vector_store import SecureVectorStore

def test_direct_query():
    """Vector store'da direkt query testi"""
    print("🔍 Vector store direkt query testi...")
    
    try:
        # Vector store başlat
        vector_store = SecureVectorStore(
            persist_directory="./data/vector_db",
            collection_name="kredi_rag_documents"
        )
        
        # Basit query - tablo belgelerini bulmaya çalış
        print("\\n📋 Basit 'tablo' query'si...")
        results = vector_store.collection.query(
            query_texts=["tablo varlık"],
            n_results=10
        )
        
        print(f"Query results: {len(results['ids'][0])} belge")
        for i, (doc_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
            metadata = results['metadatas'][0][i]
            doc_type = metadata.get('type', 'unknown')
            print(f"  {i+1}. ID: {doc_id[:50]}... Type: {doc_type}, Distance: {distance:.3f}")
        
        print("\\n📋 Tablo belgeleri için direkt metadata query...")
        table_results = vector_store.collection.query(
            query_texts=["varlık"],
            n_results=20,
            where={"type": "table"}
        )
        
        print(f"Table query results: {len(table_results['ids'][0])} belge")
        for i, (doc_id, distance) in enumerate(zip(table_results['ids'][0], table_results['distances'][0])):
            metadata = table_results['metadatas'][0][i]
            print(f"  {i+1}. ID: {doc_id[:50]}... Distance: {distance:.3f}")
            if distance < 2.0:  # Yakın sonuçları göster
                content = table_results['documents'][0][i]
                print(f"     Content preview: {content[:100]}...")
        
        print("\\n📋 Toplam belge sayısı:")
        all_docs = vector_store.collection.get()
        print(f"Toplam: {len(all_docs['ids'])}")
        
        # Tablo belgelerini say
        table_count = 0
        for metadata in all_docs['metadatas']:
            if metadata.get('type') == 'table':
                table_count += 1
        print(f"Tablo belgeleri: {table_count}")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_query() 