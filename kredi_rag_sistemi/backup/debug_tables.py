#!/usr/bin/env python3
"""
Debug table extraction process
"""

from utils.preprocessing.enhanced_pdf_processor import EnhancedPdfProcessor
import pandas as pd

def debug_tables():
    print("🔍 Tablo çıkarma süreci debug ediliyor...")
    
    processor = EnhancedPdfProcessor(use_gpu=False, table_extraction_method='auto')
    pdf_path = 'test_pdfs/KAP - Pegasus Özel Finansal Bilgiler.pdf'
    
    print("🔍 Tablolar çıkarılıyor...")
    tables = processor.extract_tables(pdf_path)
    print(f"📊 Bulunan tablo sayısı: {len(tables)}")
    
    for i, table in enumerate(tables):
        print(f"\n{'='*60}")
        print(f"TABLO {i+1}:")
        print(f"Boyut: {table.shape}")
        print(f"Başlıklar: {table.columns.tolist()}")
        print("İlk 5 satır:")
        print(table.head(5))
        
        print(f"\nTablo boş mu: {table.empty}")
        print(f"Satır sayısı: {len(table)}")
        print(f"Sütun sayısı: {len(table.columns)}")
        
        if not table.empty:
            print("\nTABLO METNİ:")
            table_text = processor.tables_to_text([table])
            print(table_text[:500] + "...")
        else:
            print("❌ Tablo boş!")
    
    # Tablo metin formatını test et
    if tables:
        print(f"\n{'='*60}")
        print("TÜM TABLOLARIN METİN FORMATI:")
        all_table_text = processor.tables_to_text(tables)
        print(all_table_text[:800] + "...")

if __name__ == "__main__":
    debug_tables() 