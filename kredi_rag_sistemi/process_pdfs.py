from app.services.pdf_processor import PDFProcessor
import json
from pathlib import Path

def main():
    # PDF işleyici oluştur
    processor = PDFProcessor("data")
    
    # Tüm PDF'leri işle
    results = processor.process_pdfs()
    
    # Sonuçları JSON olarak kaydet
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "processed_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nİşlenen PDF sayısı: {len(results)}")
    print(f"Sonuçlar şu dosyaya kaydedildi: {output_file}")
    
    # Her PDF için özet bilgileri göster
    print("\nPDF Özeti:")
    for doc in results:
        print(f"\nDosya: {doc['filename']}")
        print(f"Toplam Sayfa: {doc['total_pages']}")
        print(f"Toplam Kelime: {doc['total_words']}")

if __name__ == "__main__":
    main() 