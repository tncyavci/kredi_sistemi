#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tablo içeriği özel test scripti
Bu script tablo belgelerinin tam içeriğini gösterir
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from app.core.rag import KrediRAG

def test_table_content():
    """Tablo belgelerinin içeriğini detaylı kontrol et"""
    print("🔍 Tablo belge içeriği detaylı analizi...")
    
    try:
        # RAG sistemi başlat
        model_path = "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        vector_db_path = "./data/vector_db"
        rag = KrediRAG(model_path=model_path, vector_db_path=vector_db_path)
        
        # Tablo belgelerini ara
        results = rag.vector_store.query_documents(
            query_text="Toplam varlıklar dönen varlıklar",
            top_k=5
        )
        
        print(f"📊 Bulunan sonuç sayısı: {len(results)}")
        
        for i, result in enumerate(results):
            metadata = result.get('metadata', {})
            content = result.get('text', '')
            
            print(f"\n{'='*80}")
            print(f"SONUÇ {i+1}: {result.get('id', 'Unknown')}")
            print(f"{'='*80}")
            
            print(f"📋 TİP: {metadata.get('type', 'unknown')}")
            print(f"📅 YILLAR: {metadata.get('financial_years', [])}")
            print(f"💱 PARA BİRİMİ: {metadata.get('financial_currency', 'unknown')}")
            print(f"📊 DATA QUALITY: {metadata.get('data_quality_score', 0)}")
            print(f"🏦 TABLO TİPİ: {metadata.get('financial_table_type', 'unknown')}")
            print(f"🔢 UZAKLIK: {result.get('distance', 'unknown')}")
            
            print(f"\n📄 İÇERİK ({len(content)} karakter):")
            print("-" * 50)
            
            # İlk 2000 karakteri göster
            if len(content) > 2000:
                print(content[:2000] + "...")
                print(f"\n... [Kısaltıldı, toplam {len(content)} karakter]")
            else:
                print(content)
            
            # Eğer tablo belgesi ise özel analiz
            if metadata.get('type') == 'table':
                print(f"\n🔍 TABLO ANALİZİ:")
                
                # "Toplam varlıklar" kelimesini ara
                if "toplam varlık" in content.lower():
                    print("✅ 'Toplam varlıklar' metni bulundu")
                    
                    # Sayısal değerleri ara
                    import re
                    numbers = re.findall(r'\d{1,3}(?:\.\d{3})*(?:\.\d{3})*', content)
                    if numbers:
                        print(f"🔢 Bulunan sayılar: {numbers[:10]}")  # İlk 10 sayıyı göster
                    else:
                        print("❌ Hiç sayı bulunamadı")
                        
                    # NaN kontrolü
                    if "nan" in content.lower():
                        print("🚨 İçerikte 'NaN' değer bulundu!")
                        nan_lines = [line for line in content.split('\n') if 'nan' in line.lower()]
                        for line in nan_lines[:5]:  # İlk 5 NaN satırını göster
                            print(f"   NaN satır: {line.strip()}")
                    else:
                        print("✅ İçerikte NaN yok")
                else:
                    print("❌ 'Toplam varlıklar' metni bulunamadı")
                    
                # Spesifik sayıları ara
                target_numbers = ["1.792.338", "2.639.716", "12.687.114.838", "69.511.513.150"]
                for num in target_numbers:
                    if num in content:
                        print(f"✅ Hedef sayı bulundu: {num}")
                    else:
                        print(f"❌ Hedef sayı bulunamadı: {num}")
        
        print(f"\n{'='*80}")
        print("✅ Tablo içerik analizi tamamlandı")
        
    except Exception as e:
        print(f"❌ Hata: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_table_content() 