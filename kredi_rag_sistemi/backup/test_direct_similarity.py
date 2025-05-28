#!/usr/bin/env python3
"""
Test direct similarity calculation for table documents
"""

from models.vector_store import SecureVectorStore

def test_direct_similarity():
    print("🚀 Direct similarity testi başlıyor...")
    
    vector_store = SecureVectorStore(
        persist_directory='./data/vector_db',
        collection_name='kredi_rag_documents'
    )
    
    # ChromaDB collection'ını direkt kullan
    collection = vector_store.collection
    
    # Get table documents directly
    table_results = collection.get(
        where={"type": "table"},
        limit=10
    )
    
    table_docs = table_results.get('documents', [])
    table_metadatas = table_results.get('metadatas', [])
    table_ids = table_results.get('ids', [])
    
    print(f"🎯 Bulunan tablo sayısı: {len(table_docs)}")
    
    if not table_docs:
        print("❌ Hiç tablo belgesi bulunamadı!")
        return
    
    # Test queries with direct similarity
    test_queries = [
        "tablo",
        "finansal durum tablosu", 
        "varlık",
        "toplam varlıklar",
        "2022",
        "2023",
        "bilanço"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 50)
        
        # Direct similarity search with no filters
        try:
            results = collection.query(
                query_texts=[query],
                n_results=10,  # Get more results
                include=['documents', 'metadatas', 'distances']
            )
            
            if results and len(results['ids'][0]) > 0:
                print(f"📊 Toplam sonuç: {len(results['ids'][0])}")
                
                # Check if any tables in results
                table_found = False
                for i in range(len(results['ids'][0])):
                    doc_id = results['ids'][0][i]
                    distance = results['distances'][0][i] if 'distances' in results else 'N/A'
                    metadata = results['metadatas'][0][i] if 'metadatas' in results else {}
                    doc_type = metadata.get('type', 'unknown')
                    
                    if doc_type == 'table':
                        table_found = True
                        print(f"  🎯 TABLO BULUNDU! [{i+1}] Distance: {distance:.4f}")
                        print(f"      ID: {doc_id}")
                        
                        # Show content preview
                        if i < len(results.get('documents', [[]])[0]):
                            content = results['documents'][0][i]
                            print(f"      Content: {content[:150]}...")
                    elif i < 3:  # Show first 3 non-table results
                        print(f"  📄 [{i+1}] {doc_type} - Distance: {distance:.4f}")
                
                if not table_found:
                    print("  ❌ Tablolar bulunamadı")
                    
                    # Show distance distribution
                    distances = results.get('distances', [[]])[0]
                    if distances:
                        min_dist = min(distances)
                        max_dist = max(distances)
                        print(f"  📊 Distance aralığı: {min_dist:.4f} - {max_dist:.4f}")
                        
                        # Check for tables in all results
                        table_positions = []
                        for i, metadata in enumerate(results.get('metadatas', [[]])[0]):
                            if metadata.get('type') == 'table':
                                table_positions.append((i, distances[i]))
                        
                        if table_positions:
                            print(f"  🎯 Tablolar bulundu ancak düşük sırada:")
                            for pos, dist in table_positions:
                                print(f"    Position {pos+1}: Distance {dist:.4f}")
            else:
                print("  ❌ Hiç sonuç bulunamadı")
                
        except Exception as e:
            print(f"  ❌ Hata: {str(e)}")
    
    # Tablo içeriklerini analiz et
    print(f"\n{'='*60}")
    print("TABLO İÇERİK ANALİZİ:")
    print('='*60)
    
    for i, (doc, metadata, doc_id) in enumerate(zip(table_docs, table_metadatas, table_ids)):
        print(f"\n📊 TABLO {i+1}:")
        print(f"ID: {doc_id}")
        print(f"Uzunluk: {len(doc)} karakter")
        print(f"Kelime sayısı: {len(doc.split())}")
        
        # İçerik analizi
        lines = doc.split('\n')
        print(f"Satır sayısı: {len(lines)}")
        
        # First few lines
        print("İlk 3 satır:")
        for j, line in enumerate(lines[:3]):
            print(f"  {j+1}: {line[:100]}...")
        
        # Test manual embedding
        try:
            # Simple embedding test
            embedding_function = vector_store.embedding_function
            if hasattr(embedding_function, '__call__'):
                # Sadece ilk 500 karakteri test et
                short_content = doc[:500]
                print(f"Kısa içerik testi ({len(short_content)} karakter):")
                embedding = embedding_function([short_content])
                print(f"  Embedding boyutu: {len(embedding[0]) if embedding else 'N/A'}")
                
                # Test query similarity with short content
                test_embedding = embedding_function(["varlık"])
                if test_embedding and embedding:
                    import numpy as np
                    # Calculate cosine similarity
                    def cosine_similarity(a, b):
                        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                    
                    similarity = cosine_similarity(embedding[0], test_embedding[0])
                    print(f"  'varlık' ile benzerlik: {similarity:.4f}")
        except Exception as e:
            print(f"  Embedding test hatası: {str(e)}")

if __name__ == "__main__":
    test_direct_similarity() 