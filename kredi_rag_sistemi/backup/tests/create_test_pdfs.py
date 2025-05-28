from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os

def create_test_pdf(filename, content):
    """Create a test PDF file with given content"""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Add title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 100, "Test PDF Document")
    
    # Add content
    c.setFont("Helvetica", 12)
    y = height - 150
    for line in content.split('\n'):
        c.drawString(100, y, line)
        y -= 20
    
    c.save()

def main():
    # Create test_pdfs directory if it doesn't exist
    os.makedirs("test_pdfs", exist_ok=True)
    
    # Create different test PDFs
    test_files = [
        ("test_pdfs/simple_loan.pdf", "This is a simple loan document.\nIt contains basic loan information."),
        ("test_pdfs/mortgage_application.pdf", "Mortgage Application Form\n\nPersonal Information:\nName: John Doe\nIncome: $5000\nProperty Value: $300000"),
        ("test_pdfs/credit_card_agreement.pdf", "Credit Card Agreement\n\nTerms and Conditions:\n1. Annual Fee: $0\n2. Interest Rate: 15.99%\n3. Credit Limit: $5000"),
        ("test_pdfs/personal_loan.pdf", "Personal Loan Agreement\n\nLoan Amount: $10000\nTerm: 36 months\nInterest Rate: 12.5%"),
        ("test_pdfs/business_loan.pdf", "Business Loan Document\n\nCompany: XYZ Corp\nLoan Purpose: Equipment Purchase\nAmount: $50000")
    ]
    
    for filename, content in test_files:
        create_test_pdf(filename, content)
        print(f"Created {filename}")

if __name__ == "__main__":
    main() 