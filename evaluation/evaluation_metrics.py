import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import time
import pandas as pd
from typing import List, Dict, Any, Tuple

class RAGEvaluator:
    """
    Evaluator class for the Credit RAG System to measure various performance metrics.
    """
    
    def __init__(self, ground_truth_data=None):
        """
        Initialize the evaluator with ground truth data if available
        
        Args:
            ground_truth_data: Dictionary or dataset containing ground truth answers
        """
        self.ground_truth_data = ground_truth_data
        self.evaluation_results = {}
    
    def evaluate_retrieval(self, 
                          retrieved_docs: List[Dict], 
                          relevant_doc_ids: List[str]) -> Dict[str, float]:
        """
        Evaluate the precision and recall of document retrieval
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_doc_ids: List of relevant document IDs from ground truth
            
        Returns:
            Dictionary containing precision, recall and F1 scores
        """
        retrieved_ids = [doc.get('id') for doc in retrieved_docs]
        
        # Calculate true positives
        true_positives = len(set(retrieved_ids).intersection(set(relevant_doc_ids)))
        
        # Calculate precision, recall and F1
        precision = true_positives / len(retrieved_ids) if retrieved_ids else 0
        recall = true_positives / len(relevant_doc_ids) if relevant_doc_ids else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            'retrieval_precision': precision,
            'retrieval_recall': recall,
            'retrieval_f1': f1
        }
        
        self.evaluation_results.update(results)
        return results
    
    def evaluate_answer_accuracy(self, 
                               predictions: List[str], 
                               ground_truths: List[str],
                               method: str = 'exact_match') -> Dict[str, float]:
        """
        Evaluate the accuracy of system answers compared to ground truth
        
        Args:
            predictions: List of system-generated answers
            ground_truths: List of ground truth answers
            method: Evaluation method ('exact_match', 'fuzzy_match', or 'expert_rating')
            
        Returns:
            Dictionary containing accuracy metrics
        """
        if method == 'exact_match':
            # Calculate exact match accuracy
            matches = [1 if pred == truth else 0 for pred, truth in zip(predictions, ground_truths)]
            accuracy = sum(matches) / len(matches) if matches else 0
            
        elif method == 'fuzzy_match':
            # For fuzzy matching, we would implement a similarity score
            # This is a simplified implementation
            from difflib import SequenceMatcher
            similarities = [SequenceMatcher(None, pred, truth).ratio() 
                           for pred, truth in zip(predictions, ground_truths)]
            accuracy = sum(similarities) / len(similarities) if similarities else 0
            
        elif method == 'expert_rating':
            # In a real scenario, this would involve human expert ratings
            # For now, we'll return a placeholder
            accuracy = None
            print("Expert rating requires manual evaluation")
            
        results = {
            'answer_accuracy': accuracy,
            'evaluation_method': method
        }
        
        self.evaluation_results.update(results)
        return results
    
    def evaluate_financial_metric_extraction(self, 
                                          extracted_metrics: Dict[str, Any], 
                                          ground_truth_metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the accuracy of financial metric extraction
        
        Args:
            extracted_metrics: Dictionary of extracted financial metrics
            ground_truth_metrics: Dictionary of ground truth financial metrics
            
        Returns:
            Dictionary containing extraction accuracy metrics
        """
        # Check for each metric in ground truth
        total_metrics = len(ground_truth_metrics)
        correct_extractions = 0
        extraction_errors = []
        
        for metric_name, ground_truth_value in ground_truth_metrics.items():
            if metric_name in extracted_metrics:
                extracted_value = extracted_metrics[metric_name]
                
                # For numerical values, calculate percentage error
                if isinstance(ground_truth_value, (int, float)) and isinstance(extracted_value, (int, float)):
                    if ground_truth_value != 0:
                        error_percent = abs(extracted_value - ground_truth_value) / abs(ground_truth_value) * 100
                        # If error is less than 1%, consider it correct
                        if error_percent < 1.0:
                            correct_extractions += 1
                        extraction_errors.append((metric_name, error_percent))
                    else:
                        # Handle division by zero
                        if extracted_value == 0:
                            correct_extractions += 1
                            extraction_errors.append((metric_name, 0))
                        else:
                            extraction_errors.append((metric_name, 100))  # 100% error
                else:
                    # For non-numerical values, check exact match
                    if str(extracted_value) == str(ground_truth_value):
                        correct_extractions += 1
                        extraction_errors.append((metric_name, 0))
                    else:
                        extraction_errors.append((metric_name, 100))  # 100% error
            else:
                extraction_errors.append((metric_name, 'missing'))
        
        accuracy = correct_extractions / total_metrics if total_metrics > 0 else 0
        
        results = {
            'financial_metric_extraction_accuracy': accuracy,
            'extraction_errors': extraction_errors
        }
        
        self.evaluation_results.update(results)
        return results
    
    def evaluate_hallucination(self, 
                             system_answers: List[str], 
                             source_docs: List[Dict],
                             method: str = 'source_verification') -> Dict[str, float]:
        """
        Evaluate the system's hallucination rate
        
        Args:
            system_answers: List of system-generated answers
            source_docs: List of source documents used for answers
            method: Evaluation method
            
        Returns:
            Dictionary containing hallucination metrics
        """
        # This is a simplified implementation
        # In a real scenario, we would implement more sophisticated verification
        
        hallucination_scores = []
        
        for answer in system_answers:
            # Check if answer content can be verified in source docs
            verified = False
            for doc in source_docs:
                if doc.get('content') and any(sentence in doc['content'] for sentence in answer.split('.')):
                    verified = True
                    break
            
            hallucination_scores.append(0 if verified else 1)  # 0 = no hallucination, 1 = hallucination
        
        hallucination_rate = sum(hallucination_scores) / len(hallucination_scores) if hallucination_scores else 1
        
        results = {
            'hallucination_rate': hallucination_rate,
            'evaluation_method': method
        }
        
        self.evaluation_results.update(results)
        return results
    
    def evaluate_processing_efficiency(self, 
                                     doc_size_mb: float,
                                     processing_time_seconds: float,
                                     query_response_time_seconds: float) -> Dict[str, float]:
        """
        Evaluate the system's processing efficiency
        
        Args:
            doc_size_mb: Size of processed document in MB
            processing_time_seconds: Time taken to process document in seconds
            query_response_time_seconds: Time taken to respond to query in seconds
            
        Returns:
            Dictionary containing efficiency metrics
        """
        mb_per_second = doc_size_mb / processing_time_seconds if processing_time_seconds > 0 else 0
        
        results = {
            'processing_speed_mb_per_second': mb_per_second,
            'document_processing_time_seconds': processing_time_seconds,
            'query_response_time_seconds': query_response_time_seconds
        }
        
        self.evaluation_results.update(results)
        return results
    
    def evaluate_ocr_quality(self, 
                           extracted_text: str, 
                           ground_truth_text: str) -> Dict[str, float]:
        """
        Evaluate the OCR quality by comparing extracted text with ground truth
        
        Args:
            extracted_text: Text extracted by OCR
            ground_truth_text: Ground truth text
            
        Returns:
            Dictionary containing OCR quality metrics
        """
        # Character-level accuracy
        from difflib import SequenceMatcher
        
        # Clean texts by removing extra whitespace
        extracted_clean = ' '.join(extracted_text.split())
        ground_truth_clean = ' '.join(ground_truth_text.split())
        
        # Calculate character-level similarity
        char_similarity = SequenceMatcher(None, extracted_clean, ground_truth_clean).ratio()
        
        # Word-level accuracy
        extracted_words = extracted_clean.split()
        ground_truth_words = ground_truth_clean.split()
        
        # Count matching words
        matching_words = sum(1 for w1, w2 in zip(extracted_words, ground_truth_words) if w1 == w2)
        word_accuracy = matching_words / len(ground_truth_words) if ground_truth_words else 0
        
        results = {
            'ocr_character_similarity': char_similarity,
            'ocr_word_accuracy': word_accuracy
        }
        
        self.evaluation_results.update(results)
        return results
    
    def evaluate_table_structure(self, 
                               extracted_tables: List[pd.DataFrame], 
                               ground_truth_tables: List[pd.DataFrame]) -> Dict[str, float]:
        """
        Evaluate the system's ability to correctly understand table structures
        
        Args:
            extracted_tables: List of dataframes extracted by the system
            ground_truth_tables: List of ground truth dataframes
            
        Returns:
            Dictionary containing table structure comprehension metrics
        """
        if len(extracted_tables) != len(ground_truth_tables):
            return {'table_structure_accuracy': 0, 'error': 'Mismatched number of tables'}
        
        structure_scores = []
        content_scores = []
        
        for extracted_df, ground_truth_df in zip(extracted_tables, ground_truth_tables):
            # Structure evaluation - check columns match
            cols_match = set(extracted_df.columns) == set(ground_truth_df.columns)
            rows_match = extracted_df.shape[0] == ground_truth_df.shape[0]
            structure_score = 1.0 if cols_match and rows_match else 0.0
            
            # Content evaluation - sample comparison
            # For simplicity, we'll just check a few values
            content_match_count = 0
            sample_size = min(5, len(extracted_df)) if not extracted_df.empty else 0
            
            if sample_size > 0 and not ground_truth_df.empty:
                for i in range(sample_size):
                    row_idx = min(i, len(extracted_df)-1)
                    for col in extracted_df.columns:
                        if col in ground_truth_df.columns:
                            if extracted_df.iloc[row_idx][col] == ground_truth_df.iloc[row_idx][col]:
                                content_match_count += 1
                
                total_comparisons = sample_size * len(set(extracted_df.columns).intersection(ground_truth_df.columns))
                content_score = content_match_count / total_comparisons if total_comparisons > 0 else 0
            else:
                content_score = 0
            
            structure_scores.append(structure_score)
            content_scores.append(content_score)
        
        avg_structure_score = sum(structure_scores) / len(structure_scores) if structure_scores else 0
        avg_content_score = sum(content_scores) / len(content_scores) if content_scores else 0
        
        results = {
            'table_structure_accuracy': avg_structure_score,
            'table_content_accuracy': avg_content_score,
            'overall_table_understanding': (avg_structure_score + avg_content_score) / 2
        }
        
        self.evaluation_results.update(results)
        return results
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Return all evaluation metrics
        
        Returns:
            Dictionary containing all calculated metrics
        """
        return self.evaluation_results
    
    def generate_evaluation_report(self, output_file: str = 'evaluation_report.md') -> None:
        """
        Generate a markdown report of all evaluation metrics
        
        Args:
            output_file: Path to save the report
        """
        report = "# Credit RAG System Evaluation Report\n\n"
        
        # Group metrics by category
        categories = {
            "Retrieval Metrics": ["retrieval_precision", "retrieval_recall", "retrieval_f1"],
            "Answer Accuracy": ["answer_accuracy", "evaluation_method"],
            "Financial Extraction": ["financial_metric_extraction_accuracy"],
            "Hallucination Detection": ["hallucination_rate"],
            "Processing Efficiency": ["processing_speed_mb_per_second", "document_processing_time_seconds", 
                                      "query_response_time_seconds"],
            "OCR Quality": ["ocr_character_similarity", "ocr_word_accuracy"],
            "Table Understanding": ["table_structure_accuracy", "table_content_accuracy", "overall_table_understanding"]
        }
        
        for category, metrics in categories.items():
            report += f"## {category}\n\n"
            
            for metric in metrics:
                if metric in self.evaluation_results:
                    value = self.evaluation_results[metric]
                    if isinstance(value, float):
                        report += f"- **{metric}**: {value:.4f}\n"
                    else:
                        report += f"- **{metric}**: {value}\n"
            
            report += "\n"
        
        # Write report to file
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"Evaluation report generated at {output_file}")


# Usage example
if __name__ == "__main__":
    # This would be a real evaluation in your project
    evaluator = RAGEvaluator()
    
    # Sample evaluations
    evaluator.evaluate_retrieval(
        retrieved_docs=[{'id': 'doc1'}, {'id': 'doc2'}],
        relevant_doc_ids=['doc1', 'doc3']
    )
    
    evaluator.evaluate_answer_accuracy(
        predictions=["The total assets for 2021 were $5.7 million"],
        ground_truths=["The total assets for 2021 were $5.7 million"]
    )
    
    # Generate report
    evaluator.generate_evaluation_report() 