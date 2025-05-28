#!/usr/bin/env python3
"""
Test financial query functionality
"""

from models.vector_store import SecureVectorStore

def test_financial_query():
    print("🚀 Financial query testi başlıyor...")
    
    vector_store = SecureVectorStore(
        persist_directory='./data/vector_db',
        collection_name='kredi_rag_documents'
    )
    
    count = vector_store.get_document_count()
    print(f"📊 Toplam belge sayısı: {count}")
    
    # Financial queries to test
    test_queries = [
        {
            'query': '2022 yılında toplam varlıklar ne kadar?',
            'filters': {'years': '2022', 'only_financial': True}
        },
        {
            'query': 'varlık',
            'filters': {'only_financial': True}
        },
        {
            'query': 'bilanço',
            'filters': {'table_type': 'balance_sheet'}
        },
        {
            'query': 'finansal tablo',
            'filters': {}
        },
        {
            'query': 'gelir tablosu',
            'filters': {}
        },
        {
            'query': 'toplam varlıklar',
            'filters': {}
        }
    ]
    
    print(f"\n{'='*80}")
    print("FINANCIAL QUERY TESTLERİ:")
    print('='*80)
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case['query']
        filters = test_case['filters']
        
        print(f"\n🔍 TEST {i}: '{query}'")
        if filters:
            print(f"   Filtreler: {filters}")
        print("-" * 60)
        
        # Normal query
        normal_results = vector_store.query_documents(query, top_k=5)
        table_count_normal = sum(1 for doc in normal_results if doc.get('metadata', {}).get('type') == 'table')
        
        # Financial query
        financial_results = vector_store.query_financial_documents(
            query_text=query,
            top_k=5,
            financial_filters=filters,
            prioritize_tables=True
        )
        table_count_financial = sum(1 for doc in financial_results if doc.get('metadata', {}).get('type') == 'table')
        
        print(f"📊 Normal query: {len(normal_results)} sonuç ({table_count_normal} tablo)")
        print(f"💰 Financial query: {len(financial_results)} sonuç ({table_count_financial} tablo)")
        
        # Show top financial results
        if financial_results:
            print("🎯 En iyi financial query sonuçları:")
            for j, result in enumerate(financial_results[:3], 1):
                doc_type = result.get('metadata', {}).get('type', 'unknown')
                distance = result.get('distance', 'N/A')
                combined_score = result.get('combined_score', 'N/A')
                financial_score = result.get('financial_score', 'N/A')
                
                icon = "🎯" if doc_type == 'table' else "📄"
                print(f"  {icon} [{j}] {doc_type} - Distance: {distance:.3f}, "
                      f"Combined: {combined_score:.3f}, Financial: {financial_score:.3f}")
                print(f"      ID: {result.get('id', 'unknown')}")
                
                if doc_type == 'table':
                    content = result.get('text', '')
                    print(f"      Preview: {content[:100]}...")
        else:
            print("❌ Financial query sonuç vermedi")
    
    # Enhanced query test
    print(f"\n{'='*80}")
    print("ENHANCED QUERY TESTLERİ:")
    print('='*80)
    
    enhanced_queries = [
        "varlık asset aktif",
        "tablo table finansal",
        "2021 2022 2023 2024",
        "bilanço balance sheet",
        "toplam varlık total assets"
    ]
    
    for query in enhanced_queries:
        print(f"\n🔍 Enhanced: '{query}'")
        results = vector_store.query_documents(query, top_k=3)
        table_count = sum(1 for doc in results if doc.get('metadata', {}).get('type') == 'table')
        print(f"📊 Sonuç: {len(results)} belge ({table_count} tablo)")
        
        for j, result in enumerate(results, 1):
            doc_type = result.get('metadata', {}).get('type', 'unknown')
            distance = result.get('distance', 'N/A')
            icon = "🎯" if doc_type == 'table' else "📄"
            print(f"  {icon} [{j}] {doc_type} - Distance: {distance:.3f}")

if __name__ == "__main__":
    test_financial_query() 