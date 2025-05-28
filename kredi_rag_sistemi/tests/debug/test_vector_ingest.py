#!/usr/bin/env python3
"""
Test vector database ingestion
"""

from app.core.rag import KrediRAG
from utils.preprocessing.enhanced_pdf_processor import EnhancedPdfProcessor
import os

def test_vector_ingest():
    print("🚀 Vector database ekleme testi başlıyor...")
    
    model_path = "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    
    if not os.path.exists(model_path):
        print(f"❌ Model bulunamadı: {model_path}")
        return
    
    # RAG sistemini başlat
    print("🔧 RAG sistemi başlatılıyor...")
    rag = KrediRAG(
        model_path=model_path,
        vector_db_path="./data/vector_db",
        force_recreate_db=False  # Temiz database ile başla
    )
    
    print(f"📊 Başlangıç belge sayısı: {rag.get_document_count()}")
    
    # PDF'den belgeleri oluştur
    pdf_path = 'test_pdfs/KAP - Pegasus Özel Finansal Bilgiler.pdf'
    
    print("📝 PDF'den belgeler oluşturuluyor...")
    documents = EnhancedPdfProcessor.process_pdf_to_documents(
        pdf_path=pdf_path,
        category="finansal",
        chunk_size=1000,
        overlap=200,
        extract_tables=True,
        prioritize_tables=True,
        keep_table_context=True
    )
    
    print(f"📊 Oluşturulan belge sayısı: {len(documents)}")
    
    # Belge türlerini analiz et
    doc_types = {}
    for doc in documents:
        doc_type = doc.get('metadata', {}).get('type', 'unknown')
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    print("\n📋 Oluşturulan belge türleri:")
    for doc_type, count in doc_types.items():
        print(f"  - {doc_type}: {count} adet")
    
    # Belgeleri vector database'e ekle
    print("\n💾 Belgeler vector database'e ekleniyor...")
    rag.ingest_documents(documents)
    
    # Sonuç kontrolü
    final_count = rag.get_document_count()
    print(f"📊 Final belge sayısı: {final_count}")
    
    # Vector DB içeriğini kontrol et
    print("\n🔍 Vector database içeriği kontrol ediliyor...")
    from models.vector_store import SecureVectorStore
    
    vector_store = SecureVectorStore(
        persist_directory='./data/vector_db',
        collection_name='kredi_rag_documents'
    )
    
    # Tablo sorgusu yap
    table_docs = vector_store.query_documents('tablo', top_k=10)
    print(f"🔍 'Tablo' sorgusu sonucu: {len(table_docs)} belge")
    
    # Belge türlerini say
    retrieved_doc_types = {}
    for doc in table_docs:
        doc_type = doc.get('metadata', {}).get('type', 'unknown')
        retrieved_doc_types[doc_type] = retrieved_doc_types.get(doc_type, 0) + 1
    
    print("\n📋 Vector DB'den alınan belge türleri:")
    for doc_type, count in retrieved_doc_types.items():
        print(f"  - {doc_type}: {count} adet")
    
    # Tablo belgeleri varsa içeriğini göster
    table_docs_found = [doc for doc in table_docs if doc.get('metadata', {}).get('type') == 'table']
    if table_docs_found:
        print(f"\n✅ {len(table_docs_found)} tablo belgesi vector DB'de bulundu!")
        
        first_table = table_docs_found[0]
        print("\n📄 İlk tablo belgesi:")
        print(f"ID: {first_table.get('id', 'unknown')}")
        print(f"Metadata keys: {list(first_table.get('metadata', {}).keys())}")
        print(f"Content preview: {first_table.get('text', '')[:200]}...")
    else:
        print("\n❌ Vector DB'de hiç tablo belgesi bulunamadı!")
        
        # Alternatif sorgular dene
        alt_queries = ['finansal', 'bilanço', 'varlık', '2022']
        for query in alt_queries:
            result = vector_store.query_documents(query, top_k=3)
            table_count = sum(1 for doc in result if doc.get('metadata', {}).get('type') == 'table')
            print(f"🔍 '{query}' sorgusu: {len(result)} belge ({table_count} tablo)")

if __name__ == "__main__":
    test_vector_ingest() 