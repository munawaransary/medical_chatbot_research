"""
Data preprocessing module for Bengali medical chatbot.
Handles data cleaning, normalization, and preparation for training.
"""

import re
import logging
import unicodedata
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

logger = logging.getLogger(__name__)


class MedicalDataPreprocessor:
    """
    Preprocesses medical text data for Bengali medical chatbot training.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the data preprocessor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.processing_config = self.config.get('processing', {})
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        # Medical entity patterns
        self.medical_entities = self._load_medical_entities()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = "config/data_config.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config.get('data', {})
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return {}
    
    def _compile_patterns(self):
        """Compile regex patterns for text cleaning."""
        # HTML tags
        self.html_pattern = re.compile(r'<[^>]+>')
        
        # Extra whitespace
        self.whitespace_pattern = re.compile(r'\s+')
        
        # URLs
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        
        # Email addresses
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # Phone numbers (basic pattern)
        self.phone_pattern = re.compile(r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}')
        
        # Medical dosage patterns
        self.dosage_pattern = re.compile(r'\b\d+\s*(mg|ml|g|kg|mcg|units?)\b', re.IGNORECASE)
        
        # Bengali text normalization patterns
        self.bengali_normalize_patterns = [
            (re.compile(r'[০-৯]'), self._convert_bengali_digits),  # Bengali digits to English
            (re.compile(r'[।]'), '.'),  # Bengali full stop to English
            (re.compile(r'[‍]'), ''),   # Zero-width joiner cleanup
        ]
    
    def _load_medical_entities(self) -> Dict[str, List[str]]:
        """Load medical entity patterns for recognition."""
        return {
            'symptoms': [
                'fever', 'জ্বর', 'headache', 'মাথাব্যথা', 'cough', 'কাশি',
                'pain', 'ব্যথা', 'nausea', 'বমি বমি ভাব', 'fatigue', 'ক্লান্তি'
            ],
            'diseases': [
                'diabetes', 'ডায়াবেটিস', 'hypertension', 'উচ্চ রক্তচাপ',
                'asthma', 'হাঁপানি', 'depression', 'বিষণ্নতা'
            ],
            'medications': [
                'paracetamol', 'প্যারাসিটামল', 'aspirin', 'অ্যাসপিরিন',
                'insulin', 'ইনসুলিন', 'antibiotic', 'অ্যান্টিবায়োটিক'
            ],
            'body_parts': [
                'head', 'মাথা', 'chest', 'বুক', 'stomach', 'পেট',
                'heart', 'হৃদয়', 'lung', 'ফুসফুস', 'kidney', 'কিডনি'
            ]
        }
    
    def _convert_bengali_digits(self, match) -> str:
        """Convert Bengali digits to English digits."""
        bengali_to_english = {
            '০': '0', '১': '1', '২': '2', '৩': '3', '৪': '4',
            '৫': '5', '৬': '6', '৭': '7', '৮': '8', '৯': '9'
        }
        return bengali_to_english.get(match.group(), match.group())
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        if self.processing_config.get('cleaning', {}).get('remove_html', True):
            text = self.html_pattern.sub('', text)
        
        # Remove URLs and emails (preserve medical information)
        text = self.url_pattern.sub('', text)
        text = self.email_pattern.sub('', text)
        
        # Unicode normalization
        if self.processing_config.get('cleaning', {}).get('normalize_unicode', True):
            text = unicodedata.normalize('NFKC', text)
        
        # Bengali text normalization
        for pattern, replacement in self.bengali_normalize_patterns:
            if callable(replacement):
                text = pattern.sub(replacement, text)
            else:
                text = pattern.sub(replacement, text)
        
        # Remove extra whitespace
        if self.processing_config.get('cleaning', {}).get('remove_extra_whitespace', True):
            text = self.whitespace_pattern.sub(' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def validate_text_quality(self, text: str) -> Tuple[bool, Dict[str, Union[bool, float]]]:
        """
        Validate text quality based on various criteria.
        
        Args:
            text: Text to validate
            
        Returns:
            Tuple of (is_valid, quality_metrics)
        """
        quality_metrics = {}
        
        # Length checks
        min_length = self.processing_config.get('quality_filter', {}).get('min_length', 10)
        max_length = self.processing_config.get('quality_filter', {}).get('max_length', 1000)
        
        quality_metrics['length_valid'] = min_length <= len(text) <= max_length
        quality_metrics['text_length'] = len(text)
        
        # Language detection (basic)
        bengali_chars = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
        english_chars = sum(1 for char in text if char.isalpha() and char.isascii())
        total_chars = bengali_chars + english_chars
        
        if total_chars > 0:
            quality_metrics['bengali_ratio'] = bengali_chars / total_chars
            quality_metrics['english_ratio'] = english_chars / total_chars
        else:
            quality_metrics['bengali_ratio'] = 0.0
            quality_metrics['english_ratio'] = 0.0
        
        # Medical content detection
        medical_terms_found = 0
        for category, terms in self.medical_entities.items():
            for term in terms:
                if term.lower() in text.lower():
                    medical_terms_found += 1
        
        quality_metrics['medical_terms_count'] = medical_terms_found
        quality_metrics['has_medical_content'] = medical_terms_found > 0
        
        # Overall validity
        is_valid = (
            quality_metrics['length_valid'] and
            (quality_metrics['bengali_ratio'] > 0.1 or quality_metrics['english_ratio'] > 0.1)
        )
        
        return is_valid, quality_metrics
    
    def detect_language(self, text: str) -> str:
        """
        Detect the primary language of the text.
        
        Args:
            text: Input text
            
        Returns:
            Language code ('bn' for Bengali, 'en' for English, 'mixed' for mixed)
        """
        bengali_chars = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
        english_chars = sum(1 for char in text if char.isalpha() and char.isascii())
        total_chars = bengali_chars + english_chars
        
        if total_chars == 0:
            return 'unknown'
        
        bengali_ratio = bengali_chars / total_chars
        
        if bengali_ratio > 0.7:
            return 'bn'
        elif bengali_ratio < 0.3:
            return 'en'
        else:
            return 'mixed'
    
    def remove_duplicates(self, df: pd.DataFrame, text_columns: List[str], 
                         similarity_threshold: float = 0.9) -> pd.DataFrame:
        """
        Remove duplicate entries based on text similarity.
        
        Args:
            df: Input DataFrame
            text_columns: Columns to check for similarity
            similarity_threshold: Similarity threshold for duplicate detection
            
        Returns:
            DataFrame with duplicates removed
        """
        if not text_columns:
            return df
        
        # Simple duplicate removal based on exact matches first
        df_deduplicated = df.drop_duplicates(subset=text_columns, keep='first')
        
        logger.info(f"Removed {len(df) - len(df_deduplicated)} exact duplicates")
        
        # TODO: Implement semantic similarity-based duplicate removal
        # This would require sentence embeddings and cosine similarity calculation
        
        return df_deduplicated
    
    def create_conversation_pairs(self, df: pd.DataFrame, 
                                question_col: str, answer_col: str) -> pd.DataFrame:
        """
        Create conversation pairs from Q&A data.
        
        Args:
            df: Input DataFrame
            question_col: Column name for questions
            answer_col: Column name for answers
            
        Returns:
            DataFrame with conversation pairs
        """
        # Basic conversation pair creation
        pairs = []
        
        for _, row in df.iterrows():
            question = self.clean_text(str(row[question_col]))
            answer = self.clean_text(str(row[answer_col]))
            
            # Validate both question and answer
            q_valid, q_metrics = self.validate_text_quality(question)
            a_valid, a_metrics = self.validate_text_quality(answer)
            
            if q_valid and a_valid:
                pairs.append({
                    'input_text': question,
                    'target_text': answer,
                    'input_language': self.detect_language(question),
                    'target_language': self.detect_language(answer),
                    'input_length': len(question),
                    'target_length': len(answer),
                    'medical_terms_input': q_metrics['medical_terms_count'],
                    'medical_terms_target': a_metrics['medical_terms_count'],
                })
        
        return pd.DataFrame(pairs)
    
    def stratified_split(self, df: pd.DataFrame, 
                        stratify_columns: Optional[List[str]] = None,
                        test_size: float = 0.2, 
                        val_size: float = 0.1,
                        random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create stratified train/validation/test splits.
        
        Args:
            df: Input DataFrame
            stratify_columns: Columns to use for stratification
            test_size: Proportion of test set
            val_size: Proportion of validation set
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # If no stratification columns specified, use simple random split
        if not stratify_columns:
            # First split: train+val vs test
            train_val, test = train_test_split(
                df, test_size=test_size, random_state=random_state
            )
            
            # Second split: train vs val
            val_size_adjusted = val_size / (1 - test_size)
            train, val = train_test_split(
                train_val, test_size=val_size_adjusted, random_state=random_state
            )
            
            return train, val, test
        
        # TODO: Implement proper stratified splitting based on specified columns
        # For now, use simple random split
        logger.warning("Stratified splitting not fully implemented. Using random split.")
        return self.stratified_split(df, stratify_columns=None, 
                                   test_size=test_size, val_size=val_size, 
                                   random_state=random_state)
    
    def prepare_training_data(self, df: pd.DataFrame, 
                            input_col: str = 'input_text',
                            target_col: str = 'target_text') -> Dict[str, pd.DataFrame]:
        """
        Prepare data for training with all preprocessing steps.
        
        Args:
            df: Input DataFrame
            input_col: Column name for input text
            target_col: Column name for target text
            
        Returns:
            Dictionary with train/val/test DataFrames
        """
        logger.info("Starting data preparation pipeline...")
        
        # Step 1: Clean text
        logger.info("Cleaning text...")
        df[input_col] = df[input_col].apply(self.clean_text)
        df[target_col] = df[target_col].apply(self.clean_text)
        
        # Step 2: Quality filtering
        logger.info("Filtering by quality...")
        valid_rows = []
        for idx, row in df.iterrows():
            input_valid, _ = self.validate_text_quality(row[input_col])
            target_valid, _ = self.validate_text_quality(row[target_col])
            
            if input_valid and target_valid:
                valid_rows.append(idx)
        
        df_filtered = df.loc[valid_rows].reset_index(drop=True)
        logger.info(f"Filtered from {len(df)} to {len(df_filtered)} samples")
        
        # Step 3: Remove duplicates
        logger.info("Removing duplicates...")
        df_deduplicated = self.remove_duplicates(df_filtered, [input_col, target_col])
        
        # Step 4: Create splits
        logger.info("Creating train/val/test splits...")
        train_df, val_df, test_df = self.stratified_split(df_deduplicated)
        
        logger.info(f"Final dataset sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return {
            'train': train_df,
            'validation': val_df,
            'test': test_df,
            'full': df_deduplicated
        }
    
    def save_processed_data(self, data_dict: Dict[str, pd.DataFrame], 
                          output_dir: str = "data/processed"):
        """
        Save processed data to files.
        
        Args:
            data_dict: Dictionary of DataFrames to save
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, df in data_dict.items():
            file_path = output_path / f"{split_name}.csv"
            df.to_csv(file_path, index=False, encoding='utf-8')
            logger.info(f"Saved {split_name} data to {file_path}")
    
    def get_data_statistics(self, df: pd.DataFrame, 
                          text_columns: List[str]) -> Dict[str, Union[int, float, Dict]]:
        """
        Generate statistics about the dataset.
        
        Args:
            df: Input DataFrame
            text_columns: Text columns to analyze
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_samples': len(df),
            'columns': list(df.columns),
        }
        
        for col in text_columns:
            if col in df.columns:
                texts = df[col].astype(str)
                
                # Length statistics
                lengths = texts.str.len()
                stats[f'{col}_length'] = {
                    'mean': lengths.mean(),
                    'median': lengths.median(),
                    'min': lengths.min(),
                    'max': lengths.max(),
                    'std': lengths.std()
                }
                
                # Language distribution
                languages = texts.apply(self.detect_language)
                stats[f'{col}_languages'] = languages.value_counts().to_dict()
                
                # Medical content
                medical_content = texts.apply(
                    lambda x: any(term.lower() in x.lower() 
                                for terms in self.medical_entities.values() 
                                for term in terms)
                )
                stats[f'{col}_medical_content_ratio'] = medical_content.mean()
        
        return stats


def main():
    """Example usage of MedicalDataPreprocessor."""
    preprocessor = MedicalDataPreprocessor()
    
    # Example data
    sample_data = pd.DataFrame({
        'question': [
            "What are the symptoms of diabetes?",
            "আমার মাথাব্যথা হচ্ছে, কি করব?",
            "How to treat high blood pressure?",
            "ডায়াবেটিসের লক্ষণ কি কি?",
        ],
        'answer': [
            "Common symptoms include increased thirst, frequent urination, and fatigue.",
            "মাথাব্যথার জন্য বিশ্রাম নিন এবং প্রয়োজনে ডাক্তারের পরামর্শ নিন।",
            "Treatment includes lifestyle changes and medication as prescribed by doctor.",
            "ডায়াবেটিসের লক্ষণের মধ্যে রয়েছে অতিরিক্ত তৃষ্ণা, ঘন ঘন প্রস্রাব।",
        ]
    })
    
    print("Data Preprocessing Example:")
    print("=" * 50)
    
    # Prepare training data
    processed_data = preprocessor.prepare_training_data(sample_data, 'question', 'answer')
    
    # Show statistics
    stats = preprocessor.get_data_statistics(processed_data['full'], ['question', 'answer'])
    
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
