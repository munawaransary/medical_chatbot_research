#!/usr/bin/env python3
"""
Data download script for Bengali Medical Chatbot.
Downloads datasets from various sources including Kaggle and Hugging Face.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import requests
import pandas as pd
from datasets import load_dataset
import kaggle

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.logger import setup_logger

logger = setup_logger(__name__)


class DataDownloader:
    """Downloads and organizes datasets for the Bengali medical chatbot."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the data downloader.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            'kaggle_medical_1': {
                'source': 'kaggle',
                'dataset': 'yousefsaeedian/ai-medical-chatbot',
                'filename': 'ai_medical_chatbot.csv',
                'description': 'AI Medical Chatbot Dataset'
            },
            'kaggle_medical_2': {
                'source': 'kaggle', 
                'dataset': 'saifulislamsarfaraz/medical-chatbot-dataset',
                'filename': 'medical_chatbot_dataset.csv',
                'description': 'Medical Chatbot Dataset by Sarfaraz'
            },
            'bangla_health': {
                'source': 'huggingface',
                'dataset': 'faisal4590aziz/bangla-health-related-paraphrased-dataset',
                'filename': 'bangla_health_paraphrases.json',
                'description': 'BanglaHealth Paraphrase Dataset'
            }
        }
    
    def setup_kaggle_api(self) -> bool:
        """
        Setup Kaggle API credentials.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            # Check if kaggle.json exists
            kaggle_dir = Path.home() / '.kaggle'
            kaggle_json = kaggle_dir / 'kaggle.json'
            
            if not kaggle_json.exists():
                logger.error("Kaggle API credentials not found!")
                logger.info("Please follow these steps:")
                logger.info("1. Go to https://www.kaggle.com/account")
                logger.info("2. Click 'Create New API Token'")
                logger.info("3. Save kaggle.json to ~/.kaggle/kaggle.json")
                logger.info("4. Run: chmod 600 ~/.kaggle/kaggle.json")
                return False
            
            # Test API connection
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            
            logger.info("Kaggle API setup successful")
            return True
            
        except Exception as e:
            logger.error(f"Kaggle API setup failed: {e}")
            return False
    
    def download_kaggle_dataset(self, dataset_name: str, output_filename: str) -> bool:
        """
        Download dataset from Kaggle.
        
        Args:
            dataset_name: Kaggle dataset identifier
            output_filename: Local filename to save as
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            logger.info(f"Downloading Kaggle dataset: {dataset_name}")
            
            # Download to temporary directory
            temp_dir = self.data_dir / 'temp'
            temp_dir.mkdir(exist_ok=True)
            
            # Use kaggle API
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            
            # Download dataset
            api.dataset_download_files(
                dataset_name, 
                path=str(temp_dir),
                unzip=True
            )
            
            # Find the downloaded CSV file
            csv_files = list(temp_dir.glob('*.csv'))
            if not csv_files:
                logger.error(f"No CSV files found in downloaded dataset: {dataset_name}")
                return False
            
            # Move the first CSV file to target location
            source_file = csv_files[0]
            target_file = self.data_dir / output_filename
            
            source_file.rename(target_file)
            
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_dir)
            
            logger.info(f"Successfully downloaded: {target_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download Kaggle dataset {dataset_name}: {e}")
            return False
    
    def download_huggingface_dataset(self, dataset_name: str, output_filename: str) -> bool:
        """
        Download dataset from Hugging Face.
        
        Args:
            dataset_name: Hugging Face dataset identifier
            output_filename: Local filename to save as
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            logger.info(f"Downloading Hugging Face dataset: {dataset_name}")
            
            # Load dataset
            dataset = load_dataset(dataset_name)
            
            # Convert to pandas DataFrame
            if 'train' in dataset:
                df = dataset['train'].to_pandas()
            else:
                # Take the first split available
                split_name = list(dataset.keys())[0]
                df = dataset[split_name].to_pandas()
            
            # Save as JSON (preserving Bengali text properly)
            target_file = self.data_dir / output_filename
            df.to_json(target_file, orient='records', force_ascii=False, indent=2)
            
            logger.info(f"Successfully downloaded: {target_file}")
            logger.info(f"Dataset shape: {df.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download Hugging Face dataset {dataset_name}: {e}")
            return False
    
    def download_medical_terms_dictionary(self) -> bool:
        """
        Download or create medical terms dictionary.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Creating medical terms dictionary...")
            
            # Basic medical terms dictionary (English -> Bengali)
            medical_terms = {
                # Basic medical terms
                "doctor": "ডাক্তার",
                "patient": "রোগী",
                "hospital": "হাসপাতাল", 
                "medicine": "ওষুধ",
                "treatment": "চিকিৎসা",
                "symptoms": "লক্ষণ",
                "diagnosis": "রোগ নির্ণয়",
                "prescription": "প্রেসক্রিপশন",
                "pharmacy": "ফার্মেসি",
                "emergency": "জরুরি",
                "surgery": "অস্ত্রোপচার",
                "vaccine": "টিকা",
                "allergy": "এলার্জি",
                
                # Common symptoms
                "fever": "জ্বর",
                "headache": "মাথাব্যথা", 
                "cough": "কাশি",
                "cold": "সর্দি",
                "pain": "ব্যথা",
                "nausea": "বমি বমি ভাব",
                "fatigue": "ক্লান্তি",
                "dizziness": "মাথা ঘোরা",
                "shortness of breath": "শ্বাসকষ্ট",
                "chest pain": "বুকে ব্যথা",
                
                # Common diseases
                "diabetes": "ডায়াবেটিস",
                "hypertension": "উচ্চ রক্তচাপ",
                "heart disease": "হৃদরোগ",
                "asthma": "হাঁপানি",
                "arthritis": "বাত",
                "depression": "বিষণ্নতা",
                "anxiety": "উদ্বেগ",
                "infection": "সংক্রমণ",
                "cancer": "ক্যান্সার",
                "stroke": "স্ট্রোক",
                
                # Body parts
                "head": "মাথা",
                "chest": "বুক",
                "stomach": "পেট",
                "heart": "হৃদয়",
                "lung": "ফুসফুস",
                "kidney": "কিডনি",
                "liver": "যকৃত",
                "brain": "মস্তিষ্ক",
                "eye": "চোখ",
                "ear": "কান",
                
                # Medical procedures
                "blood test": "রক্ত পরীক্ষা",
                "x-ray": "এক্স-রে",
                "ultrasound": "আল্ট্রাসাউন্ড",
                "CT scan": "সিটি স্ক্যান",
                "MRI": "এমআরআই",
                "biopsy": "বায়োপসি",
                "checkup": "চেকআপ",
                
                # Medications
                "paracetamol": "প্যারাসিটামল",
                "aspirin": "অ্যাসপিরিন",
                "insulin": "ইনসুলিন",
                "antibiotic": "অ্যান্টিবায়োটিক",
                "vitamin": "ভিটামিন",
                
                # Medical specialties
                "cardiology": "হৃদরোগ বিশেষজ্ঞ",
                "neurology": "স্নায়ুরোগ বিশেষজ্ঞ",
                "pediatrics": "শিশুরোগ বিশেষজ্ঞ",
                "gynecology": "স্ত্রীরোগ বিশেষজ্ঞ",
                "orthopedics": "অর্থোপেডিক্স",
                "psychiatry": "মানসিক রোগ বিশেষজ্ঞ",
                
                # Time-related
                "daily": "প্রতিদিন",
                "twice a day": "দিনে দুইবার",
                "before meals": "খাবারের আগে",
                "after meals": "খাবারের পরে",
                "morning": "সকালে",
                "evening": "সন্ধ্যায়",
                "night": "রাতে",
                
                # Quantities
                "tablet": "ট্যাবলেট",
                "capsule": "ক্যাপসুল",
                "syrup": "সিরাপ",
                "injection": "ইনজেকশন",
                "drops": "ড্রপ",
                
                # Instructions
                "take": "সেবন করুন",
                "apply": "প্রয়োগ করুন",
                "avoid": "এড়িয়ে চলুন",
                "rest": "বিশ্রাম নিন",
                "drink water": "পানি পান করুন",
                "consult doctor": "ডাক্তারের পরামর্শ নিন",
            }
            
            # Save dictionary
            import json
            dict_file = self.data_dir / "medical_terms_en_bn.json"
            with open(dict_file, 'w', encoding='utf-8') as f:
                json.dump(medical_terms, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Medical terms dictionary saved: {dict_file}")
            logger.info(f"Total terms: {len(medical_terms)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create medical terms dictionary: {e}")
            return False
    
    def validate_downloaded_data(self) -> Dict[str, bool]:
        """
        Validate downloaded datasets.
        
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        for dataset_id, config in self.datasets.items():
            file_path = self.data_dir / config['filename']
            
            if not file_path.exists():
                results[dataset_id] = False
                logger.warning(f"Dataset not found: {file_path}")
                continue
            
            try:
                # Try to load and validate the file
                if file_path.suffix == '.csv':
                    df = pd.read_csv(file_path)
                    logger.info(f"{dataset_id}: {df.shape[0]} rows, {df.shape[1]} columns")
                elif file_path.suffix == '.json':
                    df = pd.read_json(file_path)
                    logger.info(f"{dataset_id}: {df.shape[0]} rows, {df.shape[1]} columns")
                
                results[dataset_id] = True
                
            except Exception as e:
                results[dataset_id] = False
                logger.error(f"Failed to validate {dataset_id}: {e}")
        
        return results
    
    def download_all(self, skip_existing: bool = True) -> Dict[str, bool]:
        """
        Download all configured datasets.
        
        Args:
            skip_existing: Skip download if file already exists
            
        Returns:
            Dictionary of download results
        """
        results = {}
        
        logger.info("Starting data download process...")
        
        # Setup Kaggle API if needed
        kaggle_datasets = [k for k, v in self.datasets.items() if v['source'] == 'kaggle']
        if kaggle_datasets and not self.setup_kaggle_api():
            logger.error("Kaggle API setup failed. Skipping Kaggle datasets.")
            for dataset_id in kaggle_datasets:
                results[dataset_id] = False
        
        # Download each dataset
        for dataset_id, config in self.datasets.items():
            logger.info(f"Processing dataset: {config['description']}")
            
            file_path = self.data_dir / config['filename']
            
            # Skip if file exists and skip_existing is True
            if skip_existing and file_path.exists():
                logger.info(f"File already exists, skipping: {file_path}")
                results[dataset_id] = True
                continue
            
            # Download based on source
            if config['source'] == 'kaggle':
                success = self.download_kaggle_dataset(
                    config['dataset'], 
                    config['filename']
                )
            elif config['source'] == 'huggingface':
                success = self.download_huggingface_dataset(
                    config['dataset'],
                    config['filename']
                )
            else:
                logger.error(f"Unknown source: {config['source']}")
                success = False
            
            results[dataset_id] = success
        
        # Download medical terms dictionary
        dict_path = self.data_dir / "medical_terms_en_bn.json"
        if not skip_existing or not dict_path.exists():
            results['medical_terms'] = self.download_medical_terms_dictionary()
        else:
            results['medical_terms'] = True
        
        # Validate all downloads
        logger.info("Validating downloaded data...")
        validation_results = self.validate_downloaded_data()
        
        # Summary
        successful = sum(results.values())
        total = len(results)
        
        logger.info(f"Download complete: {successful}/{total} successful")
        
        for dataset_id, success in results.items():
            status = "✓" if success else "✗"
            logger.info(f"  {status} {dataset_id}")
        
        return results


def main():
    """Main function for data download script."""
    parser = argparse.ArgumentParser(description="Download datasets for Bengali Medical Chatbot")
    parser.add_argument(
        "--data-dir", 
        default="data/raw",
        help="Directory to store downloaded data"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--dataset",
        choices=['kaggle_medical_1', 'kaggle_medical_2', 'bangla_health', 'all'],
        default='all',
        help="Specific dataset to download"
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = DataDownloader(args.data_dir)
    
    # Download datasets
    if args.dataset == 'all':
        results = downloader.download_all(skip_existing=not args.force)
    else:
        # Download specific dataset
        config = downloader.datasets[args.dataset]
        if config['source'] == 'kaggle':
            if not downloader.setup_kaggle_api():
                logger.error("Kaggle API setup failed")
                return 1
            success = downloader.download_kaggle_dataset(
                config['dataset'], 
                config['filename']
            )
        elif config['source'] == 'huggingface':
            success = downloader.download_huggingface_dataset(
                config['dataset'],
                config['filename']
            )
        
        results = {args.dataset: success}
    
    # Exit with error code if any downloads failed
    if not all(results.values()):
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
