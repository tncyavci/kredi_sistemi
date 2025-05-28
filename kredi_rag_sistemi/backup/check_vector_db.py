#!/usr/bin/env python3
"""
Vector database table content checker
"""

from models.vector_store import SecureVectorStore
import json

def check_vector_db():
    print("🔍 Vektör veritabanı içeriği kontrol ediliyor...")
    
    try:
        # Vektör DB'yi yükle
        vector_store = SecureVectorStore(
            persist_directory='./data/vector_db',
            collection_name='kredi_rag_documents',
            embedding_function_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        
        # Belge sayısını kontrol et
        count = vector_store.get_document_count()
        print(f"📊 Toplam belge sayısı: {count}")
        
        if count == 0:
            print("❌ Vektör veritabanı boş!")
            return
        
        # Tablo sorgusu yap
        docs = vector_store.query_documents('tablo', top_k=10)
        print(f"🔍 'Tablo' sorgusu sonucu: {len(docs)} belge")
        
        # Belge türlerini say
        doc_types = {}
        for doc in docs:
            doc_type = doc.get('metadata', {}).get('type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        print("\n📋 Belge türleri:")
        for doc_type, count in doc_types.items():
            print(f"  - {doc_type}: {count} adet")
        
        # Tablo belgeleri varsa içeriğini göster
        table_docs = [doc for doc in docs if doc.get('metadata', {}).get('type') == 'table']
        if table_docs:
            print(f"\n✅ {len(table_docs)} tablo belgesi bulundu!")
            print("\n📄 İlk tablo belgesi örneği:")
            print("-" * 60)
            print(table_docs[0]['text'][:500] + "...")
            print("-" * 60)
            
            print("\n📝 Metadata örneği:")
            metadata = table_docs[0].get('metadata', {})
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        else:
            print("\n❌ Hiç tablo belgesi bulunamadı!")
            
            # Normal belgeleri kontrol et
            print("\n📄 Normal belge örneği:")
            if docs:
                print(docs[0]['text'][:300] + "...")
        
        # Finansal terimlerle arama yap
        financial_queries = ['2022', '2023', 'varlık', 'finansal']
        for query in financial_queries:
            result = vector_store.query_documents(query, top_k=3)
            table_count = sum(1 for doc in result if doc.get('metadata', {}).get('type') == 'table')
            print(f"🔍 '{query}' sorgusu: {len(result)} belge ({table_count} tablo)")
        
    except Exception as e:
        print(f"❌ Hata: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_vector_db() 