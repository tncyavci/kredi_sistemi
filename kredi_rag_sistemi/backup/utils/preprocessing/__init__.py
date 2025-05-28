"""
PDF işleme modülü.
"""

from .enhanced_pdf_processor import EnhancedPdfProcessor

# Eski sınıf adları için tutarlılık sağlayacak alias tanımlamaları
PdfProcessor = EnhancedPdfProcessor
PDFProcessor = EnhancedPdfProcessor 