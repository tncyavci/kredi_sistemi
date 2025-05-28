# Credit RAG System

This system is a RAG (Retrieval-Augmented Generation) system that can extract information from PDF files and answer questions to accelerate credit processes for financial institutions.

## Features

- **PDF Processing**: Extracts information from financial documents, contracts, and forms
- **OCR Support**: Text recognition in scanned documents
- **Table Recognition**: Structured extraction of data from financial tables
- **GPU-Powered Processing**: GPU support for high-performance OCR and document processing
- **Caching System**: Speeds up repetitive operations by caching processed documents
- **Multiple Vector Database Support**: Compatibility with ChromaDB, Milvus, or Pinecone
- **Local LLM Integration**: Local LLM support for applications requiring privacy
- **Turkish Query Support**: Optimization for Turkish financial terms and queries

## Enhanced PDF Processing Features

- **GPU-Powered OCR**: Fast text recognition with EasyOCR with GPU support
- **Caching System**: Reuse previously processed PDFs without reprocessing
- **Higher Accuracy**: More accurate results by combining multiple OCR engines
- **Large File Support**: Ability to process large PDFs up to 500MB
- **Table Extraction Improvements**: Better recognition of different table structures
- **Multiple PDF Formats**: Support for unencrypted, scanned, and hybrid PDFs
- **Parallel Processing**: Fast processing with multi-core and threading
- **Performance Monitoring**: Monitoring processing time and resources used
- **Structured Tables**: Better recognition of relationships between data cells in tables
- **Data Normalization**: More accurate results with Turkish character and text normalization

## Installation

```bash
# Install required libraries
pip install -r requirements.txt

# Tesseract OCR installation (varies by OS)
# Ubuntu:
sudo apt-get install tesseract-ocr tesseract-ocr-tur  # Turkish language pack included
# macOS:
brew install tesseract tesseract-lang  # All languages

# For GPU support
pip install easyocr
pip install nest_asyncio  # Required for Streamlit
```

## Usage

1. Processing PDFs:

```bash
python optimize_pdf_process.py --use_gpu --chunk_size 3000 --overlap 300
```

2. Starting the web interface:

```bash
streamlit run app/streamlit_app.py
```

3. Starting the API:

```bash
uvicorn app.api:app --reload
```

## Query Examples

The system can respond to both Turkish and English PDFs with queries in both Turkish and English:

- **Turkish**: "2021 yılı için toplam aktifler ne kadardır?"
- **Turkish**: "Pegasus'un özel finansal bilgileri nelerdir?"
- **Turkish**: "Nakit akışı tablosunda en büyük gider kalemi nedir?"
- **English**: "What is the total assets for 2021?"

## Table Understanding Capability

The system is specially optimized for understanding information in tables:

- Detects table structures and understands relationships within
- Correctly matches cell values and column headers
- Can interpret financial terms and numbers
- Can compare between tables

## Performance Monitoring and Testing

To test the performance of the project:

```bash
python -m pytest tests/test_pdf_processing_performance.py -v
```

This helps determine the best configuration by comparing performance in different scenarios (CPU/GPU, cached/uncached).

## Troubleshooting

- **OCR Issues**: Text extraction accuracy may decrease if Tesseract or EasyOCR is not installed
- **GPU Issues**: If you encounter errors when using GPU, run in CPU mode with `--use_gpu false` parameter
- **Turkish Character Issues**: If you experience Turkish character issues, set the `LANG=tr_TR.UTF-8` environment variable

## License

MIT

## Contributors

- Elvin Ertuğrul
- Tuncay Avci 