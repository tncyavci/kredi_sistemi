import pandas as pd
import time
from evaluation_metrics import RAGEvaluator

def create_sample_test_data():
    """Create sample data for testing the evaluator"""
    # Sample retrieved documents
    retrieved_docs = [
        {'id': 'doc1', 'content': 'The total assets for 2021 were $5.7 million.'},
        {'id': 'doc2', 'content': 'Revenue increased by 12% in 2021.'},
        {'id': 'doc3', 'content': 'Operating expenses were $3.2 million in 2021.'}
    ]
    
    # Ground truth relevant document IDs
    relevant_doc_ids = ['doc1', 'doc3', 'doc5']
    
    # Sample system and ground truth answers
    system_answers = [
        "The total assets for 2021 were $5.7 million.",
        "Operating expenses in 2021 amounted to $3.2 million.",
        "The company's revenue increased by 15% in 2021."  # This contains hallucination
    ]
    
    ground_truth_answers = [
        "The total assets for 2021 were $5.7 million.",
        "Operating expenses in 2021 amounted to $3.2 million.",
        "The company's revenue increased by 12% in 2021."
    ]
    
    # Sample financial metrics
    extracted_metrics = {
        'total_assets_2021': 5.7,
        'operating_expenses_2021': 3.2,
        'revenue_growth_2021': 0.15,  # 15% - incorrect
        'total_liabilities_2021': 2.3
    }
    
    ground_truth_metrics = {
        'total_assets_2021': 5.7,
        'operating_expenses_2021': 3.2,
        'revenue_growth_2021': 0.12,  # 12% - correct value
        'total_liabilities_2021': 2.4,  # Slightly different from extracted
        'net_income_2021': 1.2  # Missing in extracted
    }
    
    # Sample OCR text
    ocr_extracted_text = """The company's financial position remains strong,
    with total assats of $5.7 millon and operating expensas of $3.2 million
    for the fiskal year ending December 31, 2021."""
    
    ocr_ground_truth = """The company's financial position remains strong,
    with total assets of $5.7 million and operating expenses of $3.2 million
    for the fiscal year ending December 31, 2021."""
    
    # Sample tables
    extracted_table = pd.DataFrame({
        'Metric': ['Total Assets', 'Operating Expenses', 'Revenue', 'Net Income'],
        '2020': [5.1, 3.0, 10.2, 1.0],
        '2021': [5.7, 3.2, 11.4, 1.2]
    })
    
    ground_truth_table = pd.DataFrame({
        'Metric': ['Total Assets', 'Operating Expenses', 'Revenue', 'Net Income'],
        '2020': [5.1, 3.0, 10.2, 1.0],
        '2021': [5.7, 3.2, 11.4, 1.2]
    })
    
    return {
        'retrieved_docs': retrieved_docs,
        'relevant_doc_ids': relevant_doc_ids,
        'system_answers': system_answers,
        'ground_truth_answers': ground_truth_answers,
        'extracted_metrics': extracted_metrics,
        'ground_truth_metrics': ground_truth_metrics,
        'ocr_extracted_text': ocr_extracted_text,
        'ocr_ground_truth': ocr_ground_truth,
        'extracted_tables': [extracted_table],
        'ground_truth_tables': [ground_truth_table]
    }

def run_evaluation_demo():
    """Run a demonstration of the evaluation metrics"""
    print("Running Credit RAG System Evaluation Demo")
    print("-----------------------------------------\n")
    
    # Create sample data
    test_data = create_sample_test_data()
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # 1. Evaluate Retrieval
    print("Evaluating document retrieval...")
    retrieval_results = evaluator.evaluate_retrieval(
        test_data['retrieved_docs'], 
        test_data['relevant_doc_ids']
    )
    print(f"Retrieval Precision: {retrieval_results['retrieval_precision']:.4f}")
    print(f"Retrieval Recall: {retrieval_results['retrieval_recall']:.4f}")
    print(f"Retrieval F1: {retrieval_results['retrieval_f1']:.4f}\n")
    
    # 2. Evaluate Answer Accuracy
    print("Evaluating answer accuracy...")
    # Exact match
    exact_match_results = evaluator.evaluate_answer_accuracy(
        test_data['system_answers'],
        test_data['ground_truth_answers'],
        method='exact_match'
    )
    print(f"Exact Match Accuracy: {exact_match_results['answer_accuracy']:.4f}")
    
    # Fuzzy match
    fuzzy_match_results = evaluator.evaluate_answer_accuracy(
        test_data['system_answers'],
        test_data['ground_truth_answers'],
        method='fuzzy_match'
    )
    print(f"Fuzzy Match Accuracy: {fuzzy_match_results['answer_accuracy']:.4f}\n")
    
    # 3. Evaluate Financial Metric Extraction
    print("Evaluating financial metric extraction...")
    financial_results = evaluator.evaluate_financial_metric_extraction(
        test_data['extracted_metrics'],
        test_data['ground_truth_metrics']
    )
    print(f"Financial Metric Extraction Accuracy: {financial_results['financial_metric_extraction_accuracy']:.4f}")
    print("Extraction Errors:")
    for metric, error in financial_results['extraction_errors']:
        if error == 'missing':
            print(f"  - {metric}: missing")
        else:
            print(f"  - {metric}: {error:.2f}% error")
    print()
    
    # 4. Evaluate Hallucination
    print("Evaluating hallucination...")
    hallucination_results = evaluator.evaluate_hallucination(
        test_data['system_answers'],
        test_data['retrieved_docs']
    )
    print(f"Hallucination Rate: {hallucination_results['hallucination_rate']:.4f}\n")
    
    # 5. Evaluate Processing Efficiency
    print("Evaluating processing efficiency...")
    # Simulate document processing time
    doc_size_mb = 2.5  # 2.5 MB document
    processing_time = 1.2  # 1.2 seconds
    query_time = 0.3  # 0.3 seconds
    
    efficiency_results = evaluator.evaluate_processing_efficiency(
        doc_size_mb,
        processing_time,
        query_time
    )
    print(f"Processing Speed: {efficiency_results['processing_speed_mb_per_second']:.4f} MB/s")
    print(f"Document Processing Time: {efficiency_results['document_processing_time_seconds']:.4f} s")
    print(f"Query Response Time: {efficiency_results['query_response_time_seconds']:.4f} s\n")
    
    # 6. Evaluate OCR Quality
    print("Evaluating OCR quality...")
    ocr_results = evaluator.evaluate_ocr_quality(
        test_data['ocr_extracted_text'],
        test_data['ocr_ground_truth']
    )
    print(f"OCR Character Similarity: {ocr_results['ocr_character_similarity']:.4f}")
    print(f"OCR Word Accuracy: {ocr_results['ocr_word_accuracy']:.4f}\n")
    
    # 7. Evaluate Table Structure
    print("Evaluating table structure comprehension...")
    table_results = evaluator.evaluate_table_structure(
        test_data['extracted_tables'],
        test_data['ground_truth_tables']
    )
    print(f"Table Structure Accuracy: {table_results['table_structure_accuracy']:.4f}")
    print(f"Table Content Accuracy: {table_results['table_content_accuracy']:.4f}")
    print(f"Overall Table Understanding: {table_results['overall_table_understanding']:.4f}\n")
    
    # Generate overall report
    evaluator.generate_evaluation_report("evaluation_report.md")
    print("Full evaluation report generated as 'evaluation_report.md'")

if __name__ == "__main__":
    run_evaluation_demo() 