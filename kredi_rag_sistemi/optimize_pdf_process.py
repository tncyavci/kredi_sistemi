import os
from pathlib import Path
import logging
from utils.preprocessing.enhanced_pdf_processor import EnhancedPdfProcessor
from app.core.rag import KrediRAG
import argparse
import torch
import time

# Loglama ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Komut satırı argümanlarını işle
    parser = argparse.ArgumentParser(description='Optimize edilmiş PDF işleme aracı')
    parser.add_argument('--pdf_dir', type=str, help='PDF dosyalarının bulunduğu dizin')
    parser.add_argument('--use_gpu', action='store_true', help='GPU kullanarak işleme yap')
    parser.add_argument('--chunk_size', type=int, default=3000, help='Belge parça boyutu')
    parser.add_argument('--overlap', type=int, default=300, help='Parça örtüşme miktarı')
    parser.add_argument('--batch_size', type=int, default=4, help='GPU için batch boyutu')
    args = parser.parse_args()
    
    # Get paths
    base_dir = Path(__file__).parent
    model_path = str(base_dir / "models" / "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    vector_db_path = str(base_dir / "data" / "vector_db")
    
    # PDF dizinini argümandan al veya varsayılanı kullan
    pdf_dir = args.pdf_dir if args.pdf_dir else str(base_dir / "test_pdfs")
    
    # GPU kullanılabilirliğini kontrol et
    gpu_available = torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
    use_gpu = args.use_gpu and gpu_available
    
    if args.use_gpu and not gpu_available:
        logger.warning("GPU talep edildi fakat kullanılabilir GPU bulunamadı. CPU kullanılacak.")
    
    if use_gpu:
        device_info = f"CUDA: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "Apple M1/M2 MPS"
        logger.info(f"GPU kullanılarak işleme yapılacak. {device_info}")
    else:
        logger.info("CPU kullanılarak işleme yapılacak.")
    
    print(f"PDF directory: {pdf_dir}")
    print(f"Vector DB path: {vector_db_path}")
    print(f"GPU kullanımı: {'Aktif' if use_gpu else 'Pasif'}")
    
    # Create RAG instance
    print("Initializing RAG system...")
    rag = KrediRAG(
        vector_db_path=vector_db_path,
        model_path=model_path
    )
    
    # PDF işleyici oluştur - gelişmiş ayarlarla
    pdf_processor = EnhancedPdfProcessor(
        output_dir=str(base_dir / "data" / "processed"),
        ocr_lang="tur+eng",
        use_gpu=use_gpu,
        gpu_batch_size=args.batch_size,
        memory_threshold_mb=2000,  # 2GB bellek limiti
        max_workers=os.cpu_count() or 4  # CPU çekirdek sayısı kadar worker
    )
    
    # PDF dosyalarını listele
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"PDF directory'de dosya bulunamadı: {pdf_dir}")
        return
    
    print(f"{len(pdf_files)} PDF dosyası bulundu.")
    
    # İşleme sürelerini tutmak için
    processing_times = []
    
    # Tüm PDF'leri işle
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        
        if os.path.exists(pdf_path):
            print(f"\nProcessing {pdf_file}...")
            
            try:
                # İşleme süresini ölç
                start_time = time.time()
                
                # PDF işleme - daha geniş chunk_size kullanarak daha az belge oluştur
                pdf_documents = pdf_processor.process_pdf_to_documents(
                    pdf_path=pdf_path,
                    category="finansal_bilgiler",
                    source=pdf_file,
                    chunk_size=args.chunk_size,
                    overlap=args.overlap,
                    use_ocr=True,  # OCR kullan
                    extract_tables=True,  # Tabloları çıkar
                    table_extraction_method="auto",  # Otomatik tablo çıkarma yöntemi seçimi
                    prioritize_tables=True,  # Tabloları önceliklendir
                    keep_table_context=True  # Tabloları bağlamlarıyla birlikte tut
                )
                
                # İşleme süresini kaydet
                end_time = time.time()
                process_duration = end_time - start_time
                processing_times.append((pdf_file, process_duration))
                
                print(f"Created {len(pdf_documents)} document chunks from {pdf_file}")
                print(f"İşleme süresi: {process_duration:.2f} saniye")
                
                # Belgeleri RAG sistemine aktar
                if pdf_documents:
                    print(f"Ingesting {len(pdf_documents)} documents into RAG system...")
                    rag.ingest_documents(pdf_documents)
                    print("PDF has been successfully processed and added to the vector database!")
                else:
                    print("No documents were processed.")
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")
        else:
            print(f"PDF file not found: {pdf_path}")
    
    # İşleme performansı raporu
    print("\n=== PDF İşleme Performans Raporu ===")
    for pdf_file, duration in processing_times:
        print(f"- {pdf_file}: {duration:.2f} saniye")
    
    if processing_times:
        avg_time = sum(t[1] for t in processing_times) / len(processing_times)
        print(f"\nOrtalama işleme süresi: {avg_time:.2f} saniye")
    
    # Get document count
    doc_count = rag.get_document_count()
    print(f"\nTotal document chunks in vector database: {doc_count}")

if __name__ == "__main__":
    main() 