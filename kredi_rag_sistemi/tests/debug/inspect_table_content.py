#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tablo belgelerinin tam içeriğini inspect eden script
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from app.core.rag import KrediRAG

def inspect_table_content():
    """Tablo belgelerinin tam içeriğini görüntüle"""
    print("🔍 Tablo belgelerinin tam içeriği inceleniyor...")
    
    try:
        # RAG sistemi başlat
        model_path = "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        vector_db_path = "./data/vector_db"
        rag = KrediRAG(model_path=model_path, vector_db_path=vector_db_path)
        
        # Tablo belgelerini direkt ID ile al
        table_ids = [
            "KAP - Pegasus Özel Finansal Bilgiler_table_0",
            "KAP - Pegasus Özel Finansal Bilgiler_table_1"
        ]
        
        for table_id in table_ids:
            try:
                # Belgeyi ID ile al
                result = rag.vector_store.collection.get(ids=[table_id])
                
                if result['documents']:
                    content = result['documents'][0]
                    metadata = result['metadatas'][0] if result['metadatas'] else {}
                    
                    print(f"\n{'='*80}")
                    print(f"TABLO BELGESİ: {table_id}")
                    print(f"{'='*80}")
                    
                    print(f"📊 Metadata:")
                    for key, value in metadata.items():
                        print(f"  {key}: {value}")
                    
                    print(f"\n📄 TAM İÇERİK ({len(content)} karakter):")
                    print("-" * 80)
                    print(content)
                    print("-" * 80)
                    
                    # Spesifik arama
                    print(f"\n🔍 SPESİFİK ANALİZ:")
                    
                    # Case-insensitive arama
                    content_lower = content.lower()
                    
                    # Aranacak terimler
                    search_terms = [
                        "dönen varlık",
                        "toplam dönen varlık", 
                        "1.792.338",
                        "2.639.716",
                        "12.687.114.838",
                        "69.511.513.150",
                        "toplam varlık"
                    ]
                    
                    for term in search_terms:
                        term_lower = term.lower()
                        if term_lower in content_lower:
                            print(f"✅ Bulundu: '{term}'")
                            
                            # Terimin context'ini göster
                            start_pos = content_lower.find(term_lower)
                            context_start = max(0, start_pos - 50)
                            context_end = min(len(content), start_pos + len(term) + 50)
                            context = content[context_start:context_end]
                            print(f"   Context: ...{context}...")
                        else:
                            print(f"❌ Bulunamadı: '{term}'")
                    
                    # Sayısal değerleri bul
                    import re
                    numbers = re.findall(r'\d{1,3}(?:\.\d{3})+', content)
                    if numbers:
                        print(f"\n🔢 Bulunan büyük sayılar:")
                        for num in numbers[:20]:  # İlk 20 sayıyı göster
                            print(f"   {num}")
                    
                else:
                    print(f"❌ Belge bulunamadı: {table_id}")
                    
            except Exception as e:
                print(f"❌ Hata - {table_id}: {str(e)}")
        
        print(f"\n{'='*80}")
        print("✅ Tablo içerik incelemesi tamamlandı")
        
    except Exception as e:
        print(f"❌ Genel hata: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_table_content() 