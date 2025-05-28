import sys
import os
sys.path.append('.')
from models.vector_store import SecureVectorStore

def analyze_document_sources():
    """Analyze document sources in vector database"""
    
    # Create vector store instance
    vector_store = SecureVectorStore(
        persist_directory='./data/vector_db',
        collection_name='kredi_rag_documents'
    )
    
    # Get all documents and analyze sources
    collection = vector_store.collection
    results = collection.get(include=['metadatas'])
    
    print('📋 Belge kaynakları analizi:')
    print('=' * 50)
    
    sources = {}
    categories = {}
    
    for metadata in results['metadatas']:
        source = metadata.get('source', 'Bilinmeyen')
        category = metadata.get('category', 'Bilinmeyen')
        
        if source in sources:
            sources[source] += 1
        else:
            sources[source] = 1
            
        if category in categories:
            categories[category] += 1
        else:
            categories[category] = 1
    
    print('📄 Kaynak dosyalar:')
    for source, count in sorted(sources.items()):
        print(f'  • {source}: {count} belge parçası')
    
    print(f'\n📂 Kategoriler:')
    for category, count in sorted(categories.items()):
        print(f'  • {category}: {count} belge parçası')
    
    print(f'\n📊 Toplam kaynak sayısı: {len(sources)}')
    print(f'📊 Toplam belge parçası: {sum(sources.values())}')
    
    # Show sample metadata
    if results['metadatas']:
        print('\n🔍 Örnek metadata:')
        sample_metadata = results['metadatas'][0]
        for key, value in sample_metadata.items():
            print(f'  {key}: {value}')
    
    return sources, categories

if __name__ == "__main__":
    analyze_document_sources() 