"""
Financial table processing module for credit RAG system.
Provides specialized processing for financial tables, balance sheets, and financial statements.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class FinancialTableProcessor:
    """
    Financial tables for credit evaluation and risk assessment specialized processor.
    """
    
    def __init__(self):
        """Initialize financial table processor with Turkish and English financial terms."""
        
        # Financial terms dictionary for Turkish and English
        self.financial_terms = {
            'turkish': {
                'assets': ['varlık', 'varlıklar', 'aktif', 'aktifler', 'toplam varlık'],
                'liabilities': ['yükümlülük', 'yükümlülükler', 'pasif', 'pasifler', 'borç'],
                'equity': ['özsermaye', 'öz sermaye', 'sermaye', 'equity'],
                'revenue': ['gelir', 'hasılat', 'satış', 'satışlar', 'ciro', 'revenue'],
                'expenses': ['gider', 'giderler', 'maliyet', 'maliyetler', 'expense'],
                'profit': ['kar', 'kâr', 'net kar', 'net kâr', 'profit'],
                'loss': ['zarar', 'net zarar', 'loss'],
                'cash': ['nakit', 'nakit ve nakit benzeri', 'cash'],
                'debt': ['borç', 'yükümlülük', 'kredi', 'debt'],
                'current': ['dönen', 'cari', 'kısa vadeli', 'current'],
                'non_current': ['duran', 'uzun vadeli', 'non-current'],
                'total': ['toplam', 'total'],
                'balance_sheet': ['bilanço', 'mali durum', 'finansal durum'],
                'income_statement': ['gelir tablosu', 'kar zarar tablosu'],
                'cash_flow': ['nakit akış', 'nakit akım']
            },
            'english': {
                'assets': ['assets', 'total assets', 'asset'],
                'liabilities': ['liabilities', 'liability', 'debt'],
                'equity': ['equity', 'shareholders equity', 'stockholders equity'],
                'revenue': ['revenue', 'sales', 'income', 'turnover'],
                'expenses': ['expenses', 'costs', 'expense'],
                'profit': ['profit', 'net income', 'net profit', 'earnings'],
                'loss': ['loss', 'net loss'],
                'cash': ['cash', 'cash and cash equivalents'],
                'debt': ['debt', 'borrowing', 'loan'],
                'current': ['current', 'short-term'],
                'non_current': ['non-current', 'long-term'],
                'total': ['total', 'sum'],
                'balance_sheet': ['balance sheet', 'statement of financial position'],
                'income_statement': ['income statement', 'profit and loss'],
                'cash_flow': ['cash flow', 'statement of cash flows']
            }
        }
        
        # Currency patterns
        self.currency_patterns = [
            r'(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)\s*(?:TL|₺)',  # Turkish Lira
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:\$|USD)',  # US Dollar
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:€|EUR)',  # Euro
            r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',  # General numeric
        ]
        
        # Year patterns
        self.year_pattern = r'20\d{2}'
        
    def identify_table_type(self, table: pd.DataFrame, context: str = "") -> str:
        """
        Identify the type of financial table (balance sheet, income statement, etc.)
        
        Args:
            table: DataFrame containing the table data
            context: Additional context text around the table
            
        Returns:
            String indicating the table type
        """
        table_text = self._table_to_text(table).lower()
        context_text = context.lower()
        combined_text = f"{table_text} {context_text}"
        
        # Check for balance sheet indicators
        balance_indicators = self.financial_terms['turkish']['balance_sheet'] + \
                           self.financial_terms['english']['balance_sheet'] + \
                           self.financial_terms['turkish']['assets'] + \
                           self.financial_terms['english']['assets']
        
        if any(indicator in combined_text for indicator in balance_indicators):
            return "balance_sheet"
        
        # Check for income statement indicators
        income_indicators = self.financial_terms['turkish']['income_statement'] + \
                          self.financial_terms['english']['income_statement'] + \
                          self.financial_terms['turkish']['revenue'] + \
                          self.financial_terms['english']['revenue']
        
        if any(indicator in combined_text for indicator in income_indicators):
            return "income_statement"
        
        # Check for cash flow indicators
        cash_flow_indicators = self.financial_terms['turkish']['cash_flow'] + \
                             self.financial_terms['english']['cash_flow']
        
        if any(indicator in combined_text for indicator in cash_flow_indicators):
            return "cash_flow_statement"
        
        return "financial_table"
    
    def extract_financial_values(self, table: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract financial values and key metrics from table.
        
        Args:
            table: DataFrame containing financial data
            
        Returns:
            Dictionary containing extracted financial metrics
        """
        financial_data = {
            'years': [],
            'metrics': {},
            'currency': None,
            'values': {}
        }
        
        # Extract years from headers or data
        years = self._extract_years(table)
        financial_data['years'] = years
        
        # Extract currency information
        currency = self._extract_currency(table)
        financial_data['currency'] = currency
        
        # Extract financial metrics
        metrics = self._extract_financial_metrics(table, years)
        financial_data['metrics'] = metrics
        
        return financial_data
    
    def _extract_years(self, table: pd.DataFrame) -> List[str]:
        """Extract year information from table headers and data."""
        years = []
        
        # Check column headers
        for col in table.columns:
            year_matches = re.findall(self.year_pattern, str(col))
            years.extend(year_matches)
        
        # Check table data
        for _, row in table.iterrows():
            for cell in row:
                if isinstance(cell, str):
                    year_matches = re.findall(self.year_pattern, cell)
                    years.extend(year_matches)
        
        return sorted(list(set(years)))
    
    def _extract_currency(self, table: pd.DataFrame) -> Optional[str]:
        """Extract currency information from table."""
        table_text = self._table_to_text(table)
        
        if 'TL' in table_text or '₺' in table_text:
            return 'TRY'
        elif '$' in table_text or 'USD' in table_text:
            return 'USD'
        elif '€' in table_text or 'EUR' in table_text:
            return 'EUR'
        
        return None
    
    def _extract_financial_metrics(self, table: pd.DataFrame, years: List[str]) -> Dict[str, Any]:
        """Extract specific financial metrics from table."""
        metrics = {}
        
        for idx, row in table.iterrows():
            row_text = ' '.join([str(cell) for cell in row]).lower()
            
            # Look for specific financial terms
            for category, terms in self.financial_terms['turkish'].items():
                for term in terms:
                    if term in row_text:
                        # Extract numeric values from this row
                        numeric_values = self._extract_numeric_values(row)
                        if numeric_values:
                            if category not in metrics:
                                metrics[category] = {}
                            
                            # Try to match values with years
                            for i, value in enumerate(numeric_values):
                                if i < len(years):
                                    metrics[category][years[i]] = value
                                else:
                                    metrics[category][f'value_{i}'] = value
                        break
        
        return metrics
    
    def _extract_numeric_values(self, row: pd.Series) -> List[float]:
        """Extract numeric values from a table row."""
        numeric_values = []
        
        for cell in row:
            if isinstance(cell, (int, float)) and not pd.isna(cell):
                numeric_values.append(float(cell))
            elif isinstance(cell, str):
                # Try to extract numbers from string
                # Remove currency symbols and format
                clean_text = re.sub(r'[^\d.,\-]', '', cell)
                if clean_text:
                    try:
                        # Handle different number formats
                        if ',' in clean_text and '.' in clean_text:
                            # Determine which is thousands separator
                            if clean_text.rfind(',') > clean_text.rfind('.'):
                                # Comma is decimal separator
                                clean_text = clean_text.replace('.', '').replace(',', '.')
                            else:
                                # Dot is decimal separator
                                clean_text = clean_text.replace(',', '')
                        elif ',' in clean_text:
                            # Could be thousands separator or decimal
                            parts = clean_text.split(',')
                            if len(parts) == 2 and len(parts[1]) <= 2:
                                # Likely decimal separator
                                clean_text = clean_text.replace(',', '.')
                            else:
                                # Likely thousands separator
                                clean_text = clean_text.replace(',', '')
                        
                        value = float(clean_text)
                        numeric_values.append(value)
                    except ValueError:
                        continue
        
        return numeric_values
    
    def _table_to_text(self, table: pd.DataFrame) -> str:
        """Convert table to text representation."""
        text_parts = []
        
        # Add headers
        text_parts.extend([str(col) for col in table.columns])
        
        # Add data
        for _, row in table.iterrows():
            text_parts.extend([str(cell) for cell in row])
        
        return ' '.join(text_parts)
    
    def create_financial_metadata(self, table: pd.DataFrame, table_type: str, 
                                financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create enhanced metadata for financial tables.
        
        Args:
            table: DataFrame containing the table
            table_type: Type of financial table
            financial_data: Extracted financial data
            
        Returns:
            Enhanced metadata dictionary
        """
        metadata = {
            'table_type': table_type,
            'financial_category': table_type,
            'years_covered': financial_data.get('years', []),
            'currency': financial_data.get('currency'),
            'metrics_available': list(financial_data.get('metrics', {}).keys()),
            'row_count': len(table),
            'column_count': len(table.columns),
            'has_numeric_data': self._has_numeric_data(table),
            'financial_terms_found': self._find_financial_terms(table),
            'data_quality_score': self._calculate_data_quality_score(table, financial_data)
        }
        
        return metadata
    
    def _has_numeric_data(self, table: pd.DataFrame) -> bool:
        """Check if table contains significant numeric data."""
        numeric_count = 0
        total_cells = table.size
        
        for _, row in table.iterrows():
            for cell in row:
                if isinstance(cell, (int, float)) and not pd.isna(cell):
                    numeric_count += 1
                elif isinstance(cell, str) and re.search(r'\d', cell):
                    numeric_count += 1
        
        return numeric_count / total_cells > 0.3  # At least 30% numeric content
    
    def _find_financial_terms(self, table: pd.DataFrame) -> List[str]:
        """Find financial terms present in the table."""
        table_text = self._table_to_text(table).lower()
        found_terms = []
        
        for category, terms in self.financial_terms['turkish'].items():
            for term in terms:
                if term in table_text:
                    found_terms.append(term)
        
        for category, terms in self.financial_terms['english'].items():
            for term in terms:
                if term in table_text:
                    found_terms.append(term)
        
        return list(set(found_terms))
    
    def _calculate_data_quality_score(self, table: pd.DataFrame, 
                                    financial_data: Dict[str, Any]) -> float:
        """Calculate a data quality score for the financial table."""
        score = 0.0
        
        # Check for presence of years
        if financial_data.get('years'):
            score += 0.2
        
        # Check for currency information
        if financial_data.get('currency'):
            score += 0.2
        
        # Check for financial metrics
        if financial_data.get('metrics'):
            score += 0.3
        
        # Check for numeric data
        if self._has_numeric_data(table):
            score += 0.2
        
        # Check for financial terms
        if self._find_financial_terms(table):
            score += 0.1
        
        return min(score, 1.0)
    
    def process_financial_table(self, table: pd.DataFrame, context: str = "") -> Dict[str, Any]:
        """
        Complete processing of a financial table.
        
        Args:
            table: DataFrame containing the financial table
            context: Additional context text
            
        Returns:
            Processed financial table data
        """
        # Identify table type
        table_type = self.identify_table_type(table, context)
        
        # Extract financial values
        financial_data = self.extract_financial_values(table)
        
        # Create metadata
        metadata = self.create_financial_metadata(table, table_type, financial_data)
        
        # Generate text representation
        text_representation = self._generate_financial_text(table, table_type, financial_data)
        
        return {
            'table_type': table_type,
            'financial_data': financial_data,
            'metadata': metadata,
            'text': text_representation,
            'raw_table': table.to_dict('records'),
            'processed_at': datetime.now().isoformat()
        }
    
    def _generate_financial_text(self, table: pd.DataFrame, table_type: str, 
                               financial_data: Dict[str, Any]) -> str:
        """Generate a natural language representation of the financial table."""
        text_parts = []
        
        # Add table type description
        if table_type == "balance_sheet":
            text_parts.append("Bu tablo bilanço verilerini içermektedir.")
        elif table_type == "income_statement":
            text_parts.append("Bu tablo gelir tablosu verilerini içermektedir.")
        elif table_type == "cash_flow_statement":
            text_parts.append("Bu tablo nakit akış verilerini içermektedir.")
        else:
            text_parts.append("Bu tablo finansal verileri içermektedir.")
        
        # Add years information
        if financial_data.get('years'):
            years_text = ", ".join(financial_data['years'])
            text_parts.append(f"Kapsanan yıllar: {years_text}")
        
        # Add currency information
        if financial_data.get('currency'):
            text_parts.append(f"Para birimi: {financial_data['currency']}")
        
        # Add metric information
        if financial_data.get('metrics'):
            text_parts.append("Finansal göstergeler:")
            for category, values in financial_data['metrics'].items():
                if values:
                    text_parts.append(f"- {category}: {', '.join([f'{k}: {v}' for k, v in values.items()])}")
        
        # Add raw table data in a structured format
        text_parts.append("\nTablo verileri:")
        table_text = table.to_string(index=False)
        text_parts.append(table_text)
        
        return "\n".join(text_parts) 