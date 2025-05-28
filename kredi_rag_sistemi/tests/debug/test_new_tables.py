#!/usr/bin/env python3
"""
Test new table processing system
"""

from utils.preprocessing.enhanced_pdf_processor import EnhancedPdfProcessor
from models.vector_store import SecureVectorStore
import os

def test_new_table_processing():
    print("🔧 Yeni tablo işleme sistemi test ediliyor...")
    
    # PDF'i belgeler haline çevir
    pdf_path = 'test_pdfs/KAP - Pegasus Özel Finansal Bilgiler.pdf'
    
    print("📄 PDF belgelere çevriliyor...")
    documents = EnhancedPdfProcessor.process_pdf_to_documents(
        pdf_path=pdf_path,
        category="finansal_bilgiler",
        extract_tables=True,
        prioritize_tables=False,  # Tablolar sonda
        use_gpu=False
    )
    
    print(f"📊 Toplam {len(documents)} belge oluşturuldu")
    
    # Belge türlerini say
    doc_types = {}
    for doc in documents:
        doc_type = doc['metadata'].get('type', 'unknown')
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    print("\n📋 Belge türleri:")
    for doc_type, count in doc_types.items():
        print(f"  - {doc_type}: {count} adet")
    
    # Tablo belgelerini kontrol et
    table_docs = [doc for doc in documents if doc['metadata'].get('type') == 'table']
    
    if table_docs:
        print(f"\n✅ {len(table_docs)} tablo belgesi bulundu!")
        
        for i, table_doc in enumerate(table_docs, 1):
            print(f"\n{'='*60}")
            print(f"TABLO {i} İÇERİĞİ:")
            print('='*60)
            print(table_doc['text'][:800] + "..." if len(table_doc['text']) > 800 else table_doc['text'])
            print('='*60)
            
            # Finansal veriler var mı kontrol et
            content = table_doc['text'].upper()
            financial_found = []
            if '2022' in content:
                financial_found.append('2022 yılı')
            if '2023' in content:
                financial_found.append('2023 yılı')
            if 'VARLIK' in content or 'ASSET' in content:
                financial_found.append('Varlık bilgileri')
            if 'MALIYET' in content:
                financial_found.append('Maliyet bilgileri')
            
            if financial_found:
                print(f"💰 Bulunan finansal veriler: {', '.join(financial_found)}")
            else:
                print("⚠️ Finansal veriler tespit edilemedi!")
    else:
        print("❌ Hiç tablo belgesi bulunamadı!")
    
    # Vektör veritabanına ekle
    print(f"\n📥 {len(documents)} belge vektör veritabanına ekleniyor...")
    
    vector_store = SecureVectorStore(
        persist_directory='./data/vector_db',
        collection_name='kredi_rag_documents',
        embedding_function_name='sentence-transformers/all-MiniLM-L6-v2',
        force_recreate=True  # Temiz başlangıç
    )
    
    vector_store.add_documents(documents)
    
    final_count = vector_store.get_document_count()
    print(f"✅ Vektör veritabanına {final_count} belge eklendi")
    
    # Test sorguları
    test_queries = [
        '2022 yılı toplam varlık',
        '2023 satışların maliyeti',
        'Pegasus finansal durum',
        'tablo verisi',
        'konsolide bilançosu'
    ]
    
    print(f"\n{'='*60}")
    print("TEST SORGULARI:")
    print('='*60)
    
    for query in test_queries:
        result = vector_store.query_documents(query, top_k=3)
        table_count = sum(1 for doc in result if doc.get('metadata', {}).get('type') == 'table')
        text_count = len(result) - table_count
        
        print(f"\n🔍 '{query}':")
        print(f"  📊 Bulunan belgeler: {len(result)} ({table_count} tablo, {text_count} metin)")
        
        if result and table_count > 0:
            table_doc = next((doc for doc in result if doc.get('metadata', {}).get('type') == 'table'), None)
            if table_doc:
                preview = table_doc['text'][:200].replace('\n', ' ')
                print(f"  💡 Tablo içeriği: {preview}...")
        elif result:
            preview = result[0]['text'][:200].replace('\n', ' ')
            print(f"  📄 İlk sonuç: {preview}...")
    
    print(f"\n🎉 Test tamamlandı!")

if __name__ == "__main__":
    test_new_table_processing() 