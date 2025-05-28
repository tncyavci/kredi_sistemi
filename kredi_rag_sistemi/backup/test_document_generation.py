#!/usr/bin/env python3
"""
Test document generation from PDF
"""

from utils.preprocessing.enhanced_pdf_processor import EnhancedPdfProcessor

def test_document_generation():
    print("🚀 PDF'den belge üretimi testi başlıyor...")
    
    pdf_path = 'test_pdfs/KAP - Pegasus Özel Finansal Bilgiler.pdf'
    
    print("📝 PDF belgelere dönüştürülüyor...")
    documents = EnhancedPdfProcessor.process_pdf_to_documents(
        pdf_path=pdf_path,
        category="finansal",
        chunk_size=1000,
        overlap=200,
        extract_tables=True,
        prioritize_tables=True,
        keep_table_context=True
    )
    
    print(f"📊 Toplam oluşturulan belge sayısı: {len(documents)}")
    
    # Belge türlerini analiz et
    doc_types = {}
    for doc in documents:
        doc_type = doc.get('metadata', {}).get('type', 'unknown')
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    print("\n📋 Belge türleri:")
    for doc_type, count in doc_types.items():
        print(f"  - {doc_type}: {count} adet")
    
    # Tablo belgelerini incele
    table_docs = [doc for doc in documents if doc.get('metadata', {}).get('type') == 'table']
    if table_docs:
        print(f"\n✅ {len(table_docs)} tablo belgesi oluşturuldu!")
        
        for i, table_doc in enumerate(table_docs):
            print(f"\n📄 TABLO BELGESİ {i+1}:")
            print(f"ID: {table_doc['id']}")
            
            metadata = table_doc.get('metadata', {})
            print("Metadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
            
            # İçerik önizlemesi
            content = table_doc.get('text', '')
            print(f"\nİçerik önizlemesi ({len(content)} karakter):")
            print(content[:300] + "...")
    else:
        print("\n❌ Hiç tablo belgesi oluşturulamadı!")
        
        # İlk belgeyi kontrol et
        if documents:
            print("\n📄 İlk belge örneği:")
            first_doc = documents[0]
            print(f"ID: {first_doc['id']}")
            print(f"Type: {first_doc.get('metadata', {}).get('type', 'unknown')}")
            print(f"Content: {first_doc.get('text', '')[:200]}...")

if __name__ == "__main__":
    test_document_generation() 