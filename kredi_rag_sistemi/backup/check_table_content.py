#!/usr/bin/env python3
"""
Table content detailed checker
"""

from models.vector_store import SecureVectorStore

def check_table_content():
    print("🔍 Tablo içerikleri detaylı kontrol...")
    
    vector_store = SecureVectorStore(
        persist_directory='./data/vector_db',
        collection_name='kredi_rag_documents',
        embedding_function_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    
    docs = vector_store.query_documents('tablo', top_k=5)
    table_docs = [doc for doc in docs if doc.get('metadata', {}).get('type') == 'table']
    
    if table_docs:
        print(f"📊 {len(table_docs)} tablo belgesi bulundu")
        
        for i, table_doc in enumerate(table_docs, 1):
            print(f"\n{'='*80}")
            print(f"TABLO {i} İÇERİĞİ:")
            print('='*80)
            print(table_doc['text'])
            print('='*80)
            
            # Metadata göster
            metadata = table_doc.get('metadata', {})
            print(f"Metadata: {metadata}")
            
        # Test sorguları
        test_queries = ['2022', '2023', 'toplam varlık', 'finansal durum', 'TL', 'Konsolide']
        print(f"\n{'='*80}")
        print("TEST SORGULARI:")
        print('='*80)
        
        for query in test_queries:
            result = vector_store.query_documents(query, top_k=3)
            table_count = sum(1 for doc in result if doc.get('metadata', {}).get('type') == 'table')
            text_count = len(result) - table_count
            print(f"🔍 '{query}': {len(result)} belge ({table_count} tablo, {text_count} metin)")
            
            # En alakalı belgelerin türünü göster
            if result:
                for j, doc in enumerate(result[:2], 1):
                    doc_type = doc.get('metadata', {}).get('type', 'unknown')
                    preview = doc['text'][:100].replace('\n', ' ')
                    print(f"  {j}. [{doc_type}] {preview}...")
    else:
        print("❌ Hiç tablo belgesi bulunamadı!")

if __name__ == "__main__":
    check_table_content() 