import os
from typing import List, Dict, Optional
import fitz  # PyMuPDF
import re
from datetime import datetime
import json
from pathlib import Path

class PDFProcessor:
    def __init__(self, pdf_dir: str):
        self.pdf_dir = pdf_dir
        self.text_cleaners = [
            self._remove_extra_spaces,
            self._remove_special_characters,
            self._normalize_whitespace
        ]
    
    def _remove_extra_spaces(self, text: str) -> str:
        """Fazla boşlukları temizle"""
        return re.sub(r'\s+', ' ', text)
    
    def _remove_special_characters(self, text: str) -> str:
        """Özel karakterleri temizle"""
        return re.sub(r'[^\w\s.,;:()%$€₺-]', '', text)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Boşlukları normalize et"""
        return ' '.join(text.split())
    
    def _clean_text(self, text: str) -> str:
        """Metni temizle"""
        for cleaner in self.text_cleaners:
            text = cleaner(text)
        return text.strip()
    
    def _extract_metadata(self, doc: fitz.Document) -> Dict:
        """PDF'den metadata çıkar"""
        metadata = doc.metadata
        return {
            'title': metadata.get('title', ''),
            'author': metadata.get('author', ''),
            'creation_date': metadata.get('creationDate', ''),
            'page_count': doc.page_count
        }
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """Tek bir PDF dosyasını işle"""
        try:
            doc = fitz.open(pdf_path)
            metadata = self._extract_metadata(doc)
            
            pages = []
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                cleaned_text = self._clean_text(text)
                
                if cleaned_text:  # Boş sayfaları atla
                    pages.append({
                        'page_number': page_num + 1,
                        'text': cleaned_text,
                        'word_count': len(cleaned_text.split())
                    })
            
            doc.close()
            
            return {
                'filename': os.path.basename(pdf_path),
                'metadata': metadata,
                'pages': pages,
                'total_pages': len(pages),
                'total_words': sum(page['word_count'] for page in pages),
                'processing_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"PDF işleme hatası ({pdf_path}): {str(e)}")
            return None
    
    def process_all_pdfs(self) -> List[Dict]:
        """Tüm PDF dosyalarını işle"""
        processed_docs = []
        
        for filename in os.listdir(self.pdf_dir):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_dir, filename)
                result = self.process_pdf(pdf_path)
                if result:
                    processed_docs.append(result)
        
        return processed_docs

    def save_results(self, results: List[Dict], output_dir: str = "data/processed") -> str:
        """İşlenmiş sonuçları JSON olarak kaydet"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "processed_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return str(output_file)

def main():
    # PDF işleyici oluştur
    processor = PDFProcessor("data")
    
    # Tüm PDF'leri işle
    results = processor.process_all_pdfs()
    
    # Sonuçları kaydet
    output_file = processor.save_results(results)
    
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