"""
Bengali Medical Model - Core model implementation for Bengali medical chatbot.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    T5ForConditionalGeneration, T5Tokenizer,
    GenerationConfig
)
from typing import Dict, List, Optional, Tuple, Union
import yaml
import logging
from pathlib import Path

from .cultural_adapter import CulturalAdapter

logger = logging.getLogger(__name__)


class BengaliMedicalModel(nn.Module):
    """
    Bengali Medical Chatbot Model with cultural adaptation capabilities.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Bengali medical model.
        
        Args:
            config_path: Path to model configuration file
        """
        super().__init__()
        
        self.config = self._load_config(config_path)
        self.model_config = self.config.get('model', {})
        
        # Initialize base model and tokenizer
        self._initialize_base_model()
        
        # Initialize cultural adapter
        self.cultural_adapter = CulturalAdapter(
            self.model_config.get('cultural_adapter', {})
        )
        
        # Generation configuration
        self.generation_config = self._create_generation_config()
        
        # Medical specialization modules
        self.specializations = self._initialize_specializations()
        
        logger.info(f"Initialized BengaliMedicalModel with base model: {self.model_name}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load model configuration."""
        if config_path is None:
            config_path = "config/model_config.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'model': {
                'base_model': {
                    'name': 'csebuetnlp/banglat5',
                    'tokenizer': 'csebuetnlp/banglat5'
                },
                'architecture': {
                    'max_input_length': 512,
                    'max_output_length': 256,
                    'num_beams': 4,
                    'temperature': 0.7,
                    'top_p': 0.9
                },
                'cultural_adapter': {
                    'enabled': True,
                    'adaptation_layers': 2
                },
                'specializations': {
                    'mental_health': {'enabled': True},
                    'pediatric': {'enabled': True},
                    'womens_health': {'enabled': True}
                }
            }
        }
    
    def _initialize_base_model(self):
        """Initialize the base T5 model and tokenizer."""
        base_model_config = self.model_config.get('base_model', {})
        self.model_name = base_model_config.get('name', 'csebuetnlp/banglat5')
        tokenizer_name = base_model_config.get('tokenizer', self.model_name)
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                use_fast=True
            )
            
            # Load model
            self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Add special tokens for medical domain
            special_tokens = {
                'additional_special_tokens': [
                    '<medical>', '</medical>',
                    '<symptom>', '</symptom>',
                    '<diagnosis>', '</diagnosis>',
                    '<treatment>', '</treatment>',
                    '<emergency>', '</emergency>',
                    '<cultural>', '</cultural>'
                ]
            }
            
            self.tokenizer.add_special_tokens(special_tokens)
            self.base_model.resize_token_embeddings(len(self.tokenizer))
            
            logger.info(f"Loaded base model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            # Fallback to a smaller model
            self._initialize_fallback_model()
    
    def _initialize_fallback_model(self):
        """Initialize fallback model if primary model fails."""
        logger.warning("Initializing fallback model...")
        
        try:
            self.model_name = "google/mt5-small"
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.base_model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            
            logger.info(f"Loaded fallback model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
            raise RuntimeError("Could not initialize any model")
    
    def _create_generation_config(self) -> GenerationConfig:
        """Create generation configuration."""
        arch_config = self.model_config.get('architecture', {})
        
        return GenerationConfig(
            max_length=arch_config.get('max_output_length', 256),
            min_length=10,
            num_beams=arch_config.get('num_beams', 4),
            temperature=arch_config.get('temperature', 0.7),
            top_p=arch_config.get('top_p', 0.9),
            top_k=50,
            repetition_penalty=arch_config.get('repetition_penalty', 1.2),
            length_penalty=arch_config.get('length_penalty', 1.0),
            do_sample=True,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )
    
    def _initialize_specializations(self) -> Dict:
        """Initialize medical specialization modules."""
        specializations = {}
        spec_config = self.model_config.get('specializations', {})
        
        # Mental Health Specialization
        if spec_config.get('mental_health', {}).get('enabled', False):
            specializations['mental_health'] = {
                'crisis_keywords': [
                    'আত্মহত্যা', 'suicide', 'মরে যেতে চাই', 'want to die',
                    'বাঁচতে ইচ্ছে করে না', 'don\'t want to live'
                ],
                'sensitivity_threshold': spec_config.get('mental_health', {}).get('sensitivity_threshold', 0.8)
            }
        
        # Pediatric Specialization
        if spec_config.get('pediatric', {}).get('enabled', False):
            specializations['pediatric'] = {
                'age_keywords': [
                    'শিশু', 'child', 'বাচ্চা', 'baby', 'নবজাতক', 'newborn',
                    'বছর বয়স', 'years old', 'মাস বয়স', 'months old'
                ],
                'vaccination_schedule': True
            }
        
        # Women's Health Specialization
        if spec_config.get('womens_health', {}).get('enabled', False):
            specializations['womens_health'] = {
                'keywords': [
                    'গর্ভাবস্থা', 'pregnancy', 'মাসিক', 'menstruation',
                    'স্তন', 'breast', 'জরায়ু', 'uterus'
                ],
                'cultural_sensitivity': 'high'
            }
        
        return specializations
    
    def detect_specialization(self, text: str) -> Optional[str]:
        """
        Detect medical specialization from input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Detected specialization or None
        """
        text_lower = text.lower()
        
        for spec_name, spec_config in self.specializations.items():
            keywords = spec_config.get('keywords', spec_config.get('age_keywords', spec_config.get('crisis_keywords', [])))
            
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    return spec_name
        
        return None
    
    def preprocess_input(self, text: str, specialization: Optional[str] = None) -> str:
        """
        Preprocess input text with medical and cultural context.
        
        Args:
            text: Input text
            specialization: Detected or specified specialization
            
        Returns:
            Preprocessed text with special tokens
        """
        # Add medical context token
        processed_text = f"<medical> {text} </medical>"
        
        # Add specialization context if detected
        if specialization:
            if specialization == 'mental_health':
                processed_text = f"<mental_health> {processed_text} </mental_health>"
            elif specialization == 'pediatric':
                processed_text = f"<pediatric> {processed_text} </pediatric>"
            elif specialization == 'womens_health':
                processed_text = f"<womens_health> {processed_text} </womens_health>"
        
        # Apply cultural adaptation
        processed_text = self.cultural_adapter.adapt_input(processed_text)
        
        return processed_text
    
    def postprocess_output(self, text: str, specialization: Optional[str] = None) -> str:
        """
        Postprocess model output with cultural and medical adaptations.
        
        Args:
            text: Generated text
            specialization: Medical specialization context
            
        Returns:
            Culturally adapted and medically appropriate text
        """
        # Remove special tokens
        for token in ['<medical>', '</medical>', '<mental_health>', '</mental_health>', 
                     '<pediatric>', '</pediatric>', '<womens_health>', '</womens_health>']:
            text = text.replace(token, '')
        
        # Apply cultural adaptation
        text = self.cultural_adapter.adapt_output(text, specialization)
        
        # Add medical disclaimers if needed
        text = self._add_medical_disclaimers(text, specialization)
        
        return text.strip()
    
    def _add_medical_disclaimers(self, text: str, specialization: Optional[str] = None) -> str:
        """Add appropriate medical disclaimers."""
        disclaimers = {
            'general': "\n\n⚠️ এটি শুধুমাত্র সাধারণ তথ্য। গুরুত্বপূর্ণ স্বাস্থ্য সমস্যার জন্য ডাক্তারের পরামর্শ নিন।",
            'mental_health': "\n\n⚠️ মানসিক স্বাস্থ্য সমস্যার জন্য পেশাদার সাহায্য নিন। জরুরি অবস্থায় হটলাইনে যোগাযোগ করুন।",
            'pediatric': "\n\n⚠️ শিশুর স্বাস্থ্য সমস্যার জন্য অবশ্যই শিশু বিশেষজ্ঞের পরামর্শ নিন।",
            'womens_health': "\n\n⚠️ নারী স্বাস্থ্য বিষয়ে বিশেষজ্ঞ চিকিৎসকের পরামর্শ নিন।"
        }
        
        disclaimer = disclaimers.get(specialization, disclaimers['general'])
        return text + disclaimer
    
    def generate_response(self, 
                         input_text: str, 
                         max_length: Optional[int] = None,
                         temperature: Optional[float] = None,
                         specialization: Optional[str] = None) -> str:
        """
        Generate medical response for given input.
        
        Args:
            input_text: User's medical query
            max_length: Maximum response length
            temperature: Generation temperature
            specialization: Medical specialization context
            
        Returns:
            Generated medical response
        """
        try:
            # Detect specialization if not provided
            if specialization is None:
                specialization = self.detect_specialization(input_text)
            
            # Preprocess input
            processed_input = self.preprocess_input(input_text, specialization)
            
            # Tokenize input
            inputs = self.tokenizer(
                processed_input,
                return_tensors="pt",
                max_length=self.model_config.get('architecture', {}).get('max_input_length', 512),
                truncation=True,
                padding=True
            )
            
            # Move to device
            device = next(self.base_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Update generation config if needed
            gen_config = self.generation_config
            if max_length:
                gen_config.max_length = max_length
            if temperature:
                gen_config.temperature = temperature
            
            # Generate response
            with torch.no_grad():
                outputs = self.base_model.generate(
                    **inputs,
                    generation_config=gen_config
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Postprocess output
            final_response = self.postprocess_output(response, specialization)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._get_fallback_response(specialization)
    
    def _get_fallback_response(self, specialization: Optional[str] = None) -> str:
        """Get fallback response when generation fails."""
        fallback_responses = {
            'mental_health': "আমি আপনার অনুভূতি বুঝতে পারছি। দয়া করে একজন মানসিক স্বাস্থ্য বিশেষজ্ঞের সাথে কথা বলুন।",
            'pediatric': "শিশুর স্বাস্থ্য বিষয়ে সঠিক পরামর্শের জন্য একজন শিশু বিশেষজ্ঞের কাছে যান।",
            'womens_health': "নারী স্বাস্থ্য বিষয়ে বিশেষজ্ঞ চিকিৎসকের পরামর্শ নেওয়া উত্তম।",
            'general': "আপনার প্রশ্নের জন্য একজন যোগ্য চিকিৎসকের পরামর্শ নিন।"
        }
        
        response = fallback_responses.get(specialization, fallback_responses['general'])
        return self._add_medical_disclaimers(response, specialization)
    
    def batch_generate(self, input_texts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple inputs.
        
        Args:
            input_texts: List of input texts
            **kwargs: Generation parameters
            
        Returns:
            List of generated responses
        """
        responses = []
        
        for text in input_texts:
            try:
                response = self.generate_response(text, **kwargs)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error in batch generation for text: {text[:50]}... Error: {e}")
                responses.append(self._get_fallback_response())
        
        return responses
    
    def save_model(self, save_path: str):
        """Save the model and tokenizer."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save base model and tokenizer
        self.base_model.save_pretrained(save_path / "base_model")
        self.tokenizer.save_pretrained(save_path / "tokenizer")
        
        # Save cultural adapter
        self.cultural_adapter.save(save_path / "cultural_adapter.json")
        
        # Save configuration
        with open(save_path / "model_config.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load the model and tokenizer."""
        load_path = Path(load_path)
        
        # Load base model and tokenizer
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(load_path / "base_model")
        self.tokenizer = AutoTokenizer.from_pretrained(load_path / "tokenizer")
        
        # Load cultural adapter
        self.cultural_adapter.load(load_path / "cultural_adapter.json")
        
        logger.info(f"Model loaded from {load_path}")
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'model_name': self.model_name,
            'num_parameters': sum(p.numel() for p in self.base_model.parameters()),
            'specializations': list(self.specializations.keys()),
            'cultural_adaptation': self.cultural_adapter.is_enabled(),
            'generation_config': self.generation_config.to_dict(),
            'device': str(next(self.base_model.parameters()).device)
        }


def main():
    """Example usage of BengaliMedicalModel."""
    # Initialize model
    model = BengaliMedicalModel()
    
    # Example queries
    test_queries = [
        "আমার জ্বর হয়েছে, কি করব?",
        "What are the symptoms of diabetes?",
        "আমার বাচ্চার কাশি হচ্ছে",
        "I feel very sad and hopeless",
        "গর্ভাবস্থায় কি খাওয়া উচিত?"
    ]
    
    print("Bengali Medical Model - Example Responses:")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = model.generate_response(query)
        print(f"Response: {response}")
        print("-" * 40)
    
    # Model info
    print("\nModel Information:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
