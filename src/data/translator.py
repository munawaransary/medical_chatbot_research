"""
Medical text translation module for Bengali medical chatbot.
Handles English to Bengali translation with medical terminology preservation.
"""

import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
from googletrans import Translator
from deep_translator import GoogleTranslator
import yaml

logger = logging.getLogger(__name__)


class MedicalTranslator:
    """
    Translates medical text from English to Bengali with specialized medical terminology handling.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the medical translator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.translator = GoogleTranslator(source='en', target='bn')
        self.backup_translator = Translator()
        
        # Load medical terminology dictionary
        self.medical_terms = self._load_medical_terms()
        
        # Translation cache to avoid re-translating
        self.translation_cache = {}
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 60 / self.config.get('rate_limit', 100)  # requests per minute
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = "config/data_config.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config.get('data', {}).get('translation', {})
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return {}
    
    def _load_medical_terms(self) -> Dict[str, str]:
        """Load medical terminology dictionary."""
        terms_path = self.config.get('medical_terms', {}).get('dictionary_path', 
                                                             'data/medical_terms_en_bn.json')
        
        try:
            with open(terms_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Medical terms dictionary not found: {terms_path}")
            return self._create_default_medical_terms()
    
    def _create_default_medical_terms(self) -> Dict[str, str]:
        """Create default medical terminology mappings."""
        return {
            # Common medical terms
            "doctor": "ডাক্তার",
            "patient": "রোগী", 
            "hospital": "হাসপাতাল",
            "medicine": "ওষুধ",
            "treatment": "চিকিৎসা",
            "symptoms": "লক্ষণ",
            "diagnosis": "রোগ নির্ণয়",
            "fever": "জ্বর",
            "headache": "মাথাব্যথা",
            "cough": "কাশি",
            "cold": "সর্দি",
            "pain": "ব্যথা",
            "infection": "সংক্রমণ",
            "blood pressure": "রক্তচাপ",
            "diabetes": "ডায়াবেটিস",
            "heart disease": "হৃদরোগ",
            "prescription": "প্রেসক্রিপশন",
            "pharmacy": "ফার্মেসি",
            "emergency": "জরুরি",
            "surgery": "অস্ত্রোপচার",
            "vaccine": "টিকা",
            "allergy": "এলার্জি",
            "pregnancy": "গর্ভাবস্থা",
            "mental health": "মানসিক স্বাস্থ্য",
            "depression": "বিষণ্নতা",
            "anxiety": "উদ্বেগ",
        }
    
    def _rate_limit(self):
        """Implement rate limiting for API calls."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before translation."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Handle medical abbreviations
        medical_abbrevs = {
            'Dr.': 'Doctor',
            'BP': 'Blood Pressure',
            'HR': 'Heart Rate',
            'Rx': 'Prescription',
        }
        
        for abbrev, full_form in medical_abbrevs.items():
            text = text.replace(abbrev, full_form)
        
        return text
    
    def _postprocess_translation(self, translated_text: str, original_text: str) -> str:
        """Post-process translated text with medical terminology corrections."""
        # Apply medical term mappings
        for en_term, bn_term in self.medical_terms.items():
            # Case-insensitive replacement
            translated_text = translated_text.replace(en_term.lower(), bn_term)
            translated_text = translated_text.replace(en_term.title(), bn_term)
        
        # Cultural adaptations
        cultural_adaptations = {
            "তুমি": "আপনি",  # Use respectful form
            "তোমার": "আপনার",
            "তোমাকে": "আপনাকে",
        }
        
        for informal, formal in cultural_adaptations.items():
            translated_text = translated_text.replace(informal, formal)
        
        return translated_text
    
    def translate_text(self, text: str, use_cache: bool = True) -> Tuple[str, float]:
        """
        Translate a single text from English to Bengali.
        
        Args:
            text: English text to translate
            use_cache: Whether to use translation cache
            
        Returns:
            Tuple of (translated_text, confidence_score)
        """
        if not text or not text.strip():
            return "", 0.0
        
        # Check cache first
        if use_cache and text in self.translation_cache:
            return self.translation_cache[text], 1.0
        
        try:
            # Rate limiting
            self._rate_limit()
            
            # Preprocess
            preprocessed_text = self._preprocess_text(text)
            
            # Primary translation
            translated = self.translator.translate(preprocessed_text)
            
            # Post-process
            final_translation = self._postprocess_translation(translated, text)
            
            # Cache the result
            if use_cache:
                self.translation_cache[text] = final_translation
            
            return final_translation, 0.8  # Confidence score
            
        except Exception as e:
            logger.error(f"Primary translation failed: {e}")
            
            # Fallback to backup translator
            try:
                result = self.backup_translator.translate(text, dest='bn')
                translated = result.text
                final_translation = self._postprocess_translation(translated, text)
                
                if use_cache:
                    self.translation_cache[text] = final_translation
                
                return final_translation, 0.6  # Lower confidence for backup
                
            except Exception as backup_error:
                logger.error(f"Backup translation also failed: {backup_error}")
                return text, 0.0  # Return original text as fallback
    
    def translate_batch(self, texts: List[str], batch_size: int = 100) -> List[Tuple[str, float]]:
        """
        Translate a batch of texts.
        
        Args:
            texts: List of English texts to translate
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of (translated_text, confidence_score) tuples
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch:
                translated, confidence = self.translate_text(text)
                batch_results.append((translated, confidence))
            
            results.extend(batch_results)
            
            # Progress logging
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Translated {i + len(batch)}/{len(texts)} texts")
        
        return results
    
    def translate_dataframe(self, df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
        """
        Translate specified columns in a DataFrame.
        
        Args:
            df: Input DataFrame
            text_columns: List of column names to translate
            
        Returns:
            DataFrame with translated columns (suffixed with '_bn')
        """
        result_df = df.copy()
        
        for column in text_columns:
            if column not in df.columns:
                logger.warning(f"Column '{column}' not found in DataFrame")
                continue
            
            logger.info(f"Translating column: {column}")
            
            # Get texts to translate
            texts = df[column].fillna("").astype(str).tolist()
            
            # Translate
            translations = self.translate_batch(texts)
            
            # Add to DataFrame
            result_df[f"{column}_bn"] = [trans[0] for trans in translations]
            result_df[f"{column}_bn_confidence"] = [trans[1] for trans in translations]
        
        return result_df
    
    def back_translate(self, bengali_text: str) -> Tuple[str, float]:
        """
        Perform back-translation for quality assessment.
        
        Args:
            bengali_text: Bengali text to back-translate to English
            
        Returns:
            Tuple of (back_translated_text, quality_score)
        """
        try:
            # Rate limiting
            self._rate_limit()
            
            # Back-translate to English
            back_translator = GoogleTranslator(source='bn', target='en')
            back_translated = back_translator.translate(bengali_text)
            
            return back_translated, 0.8
            
        except Exception as e:
            logger.error(f"Back-translation failed: {e}")
            return "", 0.0
    
    def save_cache(self, cache_path: str = "data/translation_cache.json"):
        """Save translation cache to file."""
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.translation_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"Translation cache saved to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def load_cache(self, cache_path: str = "data/translation_cache.json"):
        """Load translation cache from file."""
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                self.translation_cache = json.load(f)
            logger.info(f"Translation cache loaded from {cache_path}")
        except FileNotFoundError:
            logger.info("No existing cache found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")


def main():
    """Example usage of MedicalTranslator."""
    translator = MedicalTranslator()
    
    # Example translations
    test_texts = [
        "What are the symptoms of diabetes?",
        "I have a severe headache and fever.",
        "Please consult a doctor immediately.",
        "Take this medicine twice a day after meals."
    ]
    
    print("Medical Translation Examples:")
    print("=" * 50)
    
    for text in test_texts:
        translated, confidence = translator.translate_text(text)
        print(f"EN: {text}")
        print(f"BN: {translated}")
        print(f"Confidence: {confidence:.2f}")
        print("-" * 30)


if __name__ == "__main__":
    main()
