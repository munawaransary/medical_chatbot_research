"""
Cultural Adapter - Handles cultural adaptation for Bengali medical chatbot.
"""

import json
import re
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class CulturalAdapter:
    """
    Handles cultural adaptation for Bengali medical communication.
    Ensures respectful, culturally appropriate medical advice.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize cultural adapter.
        
        Args:
            config: Cultural adaptation configuration
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        
        # Cultural adaptation rules
        self.respect_forms = self._load_respect_forms()
        self.medical_honorifics = self._load_medical_honorifics()
        self.cultural_terms = self._load_cultural_terms()
        self.sensitive_topics = self._load_sensitive_topics()
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        logger.info("Cultural adapter initialized")
    
    def _load_respect_forms(self) -> Dict[str, str]:
        """Load respectful forms of address."""
        return {
            # Informal to formal pronouns
            'তুমি': 'আপনি',
            'তোমার': 'আপনার',
            'তোমাকে': 'আপনাকে',
            'তোমায়': 'আপনাকে',
            'তোর': 'আপনার',
            'তোকে': 'আপনাকে',
            
            # Verb forms (informal to formal)
            'করো': 'করুন',
            'যাও': 'যান',
            'এসো': 'আসুন',
            'বলো': 'বলুন',
            'নাও': 'নিন',
            'দেখো': 'দেখুন',
            'শোনো': 'শুনুন',
            'খাও': 'খান',
            'পড়ো': 'পড়ুন',
            
            # Common informal commands to formal
            'কর': 'করুন',
            'যা': 'যান',
            'আয়': 'আসুন',
            'বল': 'বলুন',
            'নে': 'নিন',
            'দেখ': 'দেখুন',
            'শোন': 'শুনুন',
            'খা': 'খান',
            'পড়': 'পড়ুন',
        }
    
    def _load_medical_honorifics(self) -> Dict[str, str]:
        """Load medical honorifics and respectful terms."""
        return {
            'ডাক্তার': 'ডাক্তার সাহেব',
            'চিকিৎসক': 'চিকিৎসক মহোদয়',
            'নার্স': 'নার্স বোন',
            'রোগী': 'রোগী ভাই/বোন',
            
            # Professional titles
            'doctor': 'ডাক্তার সাহেব',
            'physician': 'চিকিৎসক মহোদয়',
            'specialist': 'বিশেষজ্ঞ ডাক্তার',
            'consultant': 'পরামর্শদাতা চিকিৎসক',
        }
    
    def _load_cultural_terms(self) -> Dict[str, str]:
        """Load culturally appropriate medical terms."""
        return {
            # Body parts with cultural sensitivity
            'private parts': 'গোপনাঙ্গ',
            'genitals': 'প্রজনন অঙ্গ',
            'breast': 'স্তন',
            'chest': 'বুক',
            
            # Sensitive medical conditions
            'mental illness': 'মানসিক স্বাস্থ্য সমস্যা',
            'depression': 'মানসিক অবসাদ',
            'anxiety': 'মানসিক উদ্বেগ',
            'suicide': 'আত্মহত্যার চিন্তা',
            
            # Pregnancy and reproductive health
            'pregnancy': 'গর্ভাবস্থা',
            'menstruation': 'মাসিক',
            'fertility': 'প্রজনন ক্ষমতা',
            'contraception': 'জন্মনিয়ন্ত্রণ',
            
            # Cultural practices
            'traditional medicine': 'ঐতিহ্যবাহী চিকিৎসা',
            'herbal medicine': 'ভেষজ চিকিৎসা',
            'home remedy': 'ঘরোয়া চিকিৎসা',
        }
    
    def _load_sensitive_topics(self) -> Dict[str, Dict]:
        """Load sensitive topics and their handling guidelines."""
        return {
            'mental_health': {
                'keywords': ['আত্মহত্যা', 'suicide', 'depression', 'বিষণ্নতা'],
                'approach': 'supportive',
                'referral_required': True,
                'crisis_response': True
            },
            'reproductive_health': {
                'keywords': ['গর্ভাবস্থা', 'pregnancy', 'মাসিক', 'menstruation'],
                'approach': 'respectful',
                'gender_sensitivity': True,
                'privacy_emphasis': True
            },
            'pediatric': {
                'keywords': ['শিশু', 'child', 'বাচ্চা', 'baby'],
                'approach': 'careful',
                'parental_involvement': True,
                'age_appropriate': True
            },
            'elderly_care': {
                'keywords': ['বয়স্ক', 'elderly', 'বৃদ্ধ', 'old age'],
                'approach': 'respectful',
                'family_involvement': True,
                'dignity_preservation': True
            }
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient text processing."""
        # Pattern for informal pronouns and verbs
        informal_patterns = []
        for informal, formal in self.respect_forms.items():
            # Word boundary patterns to avoid partial matches
            pattern = rf'\b{re.escape(informal)}\b'
            informal_patterns.append((re.compile(pattern, re.IGNORECASE), formal))
        
        self.informal_patterns = informal_patterns
        
        # Pattern for medical terms that need honorifics
        honorific_patterns = []
        for term, honorific in self.medical_honorifics.items():
            pattern = rf'\b{re.escape(term)}\b'
            honorific_patterns.append((re.compile(pattern, re.IGNORECASE), honorific))
        
        self.honorific_patterns = honorific_patterns
    
    def is_enabled(self) -> bool:
        """Check if cultural adaptation is enabled."""
        return self.enabled
    
    def detect_sensitive_topic(self, text: str) -> Optional[str]:
        """
        Detect sensitive topics in the text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Detected sensitive topic or None
        """
        text_lower = text.lower()
        
        for topic, config in self.sensitive_topics.items():
            keywords = config.get('keywords', [])
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    return topic
        
        return None
    
    def adapt_input(self, text: str) -> str:
        """
        Adapt input text for cultural appropriateness.
        
        Args:
            text: Input text to adapt
            
        Returns:
            Culturally adapted text
        """
        if not self.enabled:
            return text
        
        adapted_text = text
        
        # Convert informal forms to formal
        for pattern, formal in self.informal_patterns:
            adapted_text = pattern.sub(formal, adapted_text)
        
        # Add cultural context markers for sensitive topics
        sensitive_topic = self.detect_sensitive_topic(adapted_text)
        if sensitive_topic:
            adapted_text = f"<cultural:{sensitive_topic}> {adapted_text} </cultural:{sensitive_topic}>"
        
        return adapted_text
    
    def adapt_output(self, text: str, specialization: Optional[str] = None) -> str:
        """
        Adapt output text for cultural appropriateness.
        
        Args:
            text: Generated text to adapt
            specialization: Medical specialization context
            
        Returns:
            Culturally adapted output text
        """
        if not self.enabled:
            return text
        
        adapted_text = text
        
        # Remove cultural context markers
        adapted_text = re.sub(r'<cultural:[^>]*>', '', adapted_text)
        adapted_text = re.sub(r'</cultural:[^>]*>', '', adapted_text)
        
        # Apply respectful forms
        for pattern, formal in self.informal_patterns:
            adapted_text = pattern.sub(formal, adapted_text)
        
        # Add appropriate honorifics
        for pattern, honorific in self.honorific_patterns:
            adapted_text = pattern.sub(honorific, adapted_text)
        
        # Apply specialization-specific adaptations
        adapted_text = self._apply_specialization_adaptations(adapted_text, specialization)
        
        # Add cultural sensitivity notes if needed
        adapted_text = self._add_cultural_sensitivity_notes(adapted_text, specialization)
        
        return adapted_text
    
    def _apply_specialization_adaptations(self, text: str, specialization: Optional[str]) -> str:
        """Apply specialization-specific cultural adaptations."""
        if not specialization:
            return text
        
        if specialization == 'mental_health':
            # Add supportive language
            if any(word in text.lower() for word in ['depression', 'বিষণ্নতা', 'sad', 'দুঃখ']):
                text = self._add_mental_health_support(text)
        
        elif specialization == 'womens_health':
            # Add privacy and respect emphasis
            text = self._add_privacy_emphasis(text)
        
        elif specialization == 'pediatric':
            # Add parental involvement emphasis
            text = self._add_parental_guidance(text)
        
        return text
    
    def _add_mental_health_support(self, text: str) -> str:
        """Add supportive language for mental health topics."""
        supportive_phrases = [
            "আপনার অনুভূতি স্বাভাবিক এবং আপনি একা নন।",
            "মানসিক স্বাস্থ্য শারীরিক স্বাস্থ্যের মতোই গুরুত্বপূর্ণ।",
            "সাহায্য চাওয়া শক্তির লক্ষণ, দুর্বলতার নয়।"
        ]
        
        # Add a supportive phrase if mental health keywords are detected
        if any(word in text.lower() for word in ['depression', 'বিষণ্নতা', 'anxiety', 'উদ্বেগ']):
            text = f"{text}\n\n💙 {supportive_phrases[0]}"
        
        return text
    
    def _add_privacy_emphasis(self, text: str) -> str:
        """Add privacy emphasis for women's health topics."""
        privacy_note = "\n\n🔒 আপনার গোপনীয়তা আমাদের কাছে গুরুত্বপূর্ণ। বিশ্বস্ত চিকিৎসকের সাথে আলোচনা করুন।"
        
        sensitive_keywords = ['pregnancy', 'গর্ভাবস্থা', 'menstruation', 'মাসিক', 'reproductive', 'প্রজনন']
        if any(keyword in text.lower() for keyword in sensitive_keywords):
            text += privacy_note
        
        return text
    
    def _add_parental_guidance(self, text: str) -> str:
        """Add parental guidance for pediatric topics."""
        parental_note = "\n\n👨‍👩‍👧‍👦 শিশুর স্বাস্থ্য বিষয়ে সিদ্ধান্ত নেওয়ার সময় অভিভাবকদের সাথে আলোচনা করুন।"
        
        pediatric_keywords = ['child', 'শিশু', 'baby', 'বাচ্চা', 'kid', 'বাচ্চা']
        if any(keyword in text.lower() for keyword in pediatric_keywords):
            text += parental_note
        
        return text
    
    def _add_cultural_sensitivity_notes(self, text: str, specialization: Optional[str]) -> str:
        """Add cultural sensitivity notes based on content."""
        sensitive_topic = self.detect_sensitive_topic(text)
        
        if sensitive_topic == 'mental_health':
            text += "\n\n🤝 মানসিক স্বাস্থ্য সেবা নেওয়া সামাজিকভাবে গ্রহণযোগ্য এবং প্রয়োজনীয়।"
        
        elif sensitive_topic == 'reproductive_health':
            text += "\n\n🌸 নারী স্বাস্থ্য বিষয়ে খোলামেলা আলোচনা স্বাস্থ্যকর জীবনের জন্য জরুরি।"
        
        return text
    
    def get_cultural_guidelines(self, specialization: str) -> Dict[str, str]:
        """
        Get cultural guidelines for specific medical specialization.
        
        Args:
            specialization: Medical specialization
            
        Returns:
            Cultural guidelines dictionary
        """
        guidelines = {
            'mental_health': {
                'approach': 'সহানুভূতিশীল এবং বিচারহীন',
                'language': 'সহায়ক এবং আশাব্যঞ্জক',
                'referral': 'মানসিক স্বাস্থ্য বিশেষজ্ঞের কাছে পাঠান',
                'crisis': 'জরুরি অবস্থায় হটলাইন নম্বর প্রদান করুন'
            },
            'womens_health': {
                'approach': 'সম্মানজনক এবং গোপনীয়তা রক্ষাকারী',
                'language': 'সংবেদনশীল এবং সহায়ক',
                'referral': 'নারী স্বাস্থ্য বিশেষজ্ঞের কাছে পাঠান',
                'privacy': 'গোপনীয়তার গুরুত্ব তুলে ধরুন'
            },
            'pediatric': {
                'approach': 'যত্নশীল এবং অভিভাবক-কেন্দ্রিক',
                'language': 'সহজ এবং বোধগম্য',
                'referral': 'শিশু বিশেষজ্ঞের কাছে পাঠান',
                'involvement': 'অভিভাবকদের সম্পৃক্ততা নিশ্চিত করুন'
            },
            'general': {
                'approach': 'সম্মানজনক এবং সহায়ক',
                'language': 'স্পষ্ট এবং বোধগম্য',
                'referral': 'যোগ্য চিকিৎসকের কাছে পাঠান',
                'disclaimer': 'চিকিৎসা দাবিত্যাগ অন্তর্ভুক্ত করুন'
            }
        }
        
        return guidelines.get(specialization, guidelines['general'])
    
    def validate_cultural_appropriateness(self, text: str) -> Tuple[bool, List[str]]:
        """
        Validate if text is culturally appropriate.
        
        Args:
            text: Text to validate
            
        Returns:
            Tuple of (is_appropriate, list_of_issues)
        """
        issues = []
        
        # Check for informal language
        for pattern, _ in self.informal_patterns:
            if pattern.search(text):
                issues.append("Contains informal language that should be formal")
                break
        
        # Check for missing honorifics
        medical_terms = ['ডাক্তার', 'চিকিৎসক', 'doctor', 'physician']
        for term in medical_terms:
            if term in text.lower() and f"{term} সাহেব" not in text and f"{term} মহোদয়" not in text:
                issues.append(f"Missing honorific for medical professional: {term}")
        
        # Check for sensitive topic handling
        sensitive_topic = self.detect_sensitive_topic(text)
        if sensitive_topic:
            topic_config = self.sensitive_topics[sensitive_topic]
            if topic_config.get('referral_required', False):
                if 'বিশেষজ্ঞ' not in text and 'specialist' not in text:
                    issues.append(f"Missing specialist referral for sensitive topic: {sensitive_topic}")
        
        is_appropriate = len(issues) == 0
        return is_appropriate, issues
    
    def save(self, file_path: str):
        """Save cultural adapter configuration."""
        config_data = {
            'enabled': self.enabled,
            'respect_forms': self.respect_forms,
            'medical_honorifics': self.medical_honorifics,
            'cultural_terms': self.cultural_terms,
            'sensitive_topics': self.sensitive_topics
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Cultural adapter saved to {file_path}")
    
    def load(self, file_path: str):
        """Load cultural adapter configuration."""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        self.enabled = config_data.get('enabled', True)
        self.respect_forms = config_data.get('respect_forms', {})
        self.medical_honorifics = config_data.get('medical_honorifics', {})
        self.cultural_terms = config_data.get('cultural_terms', {})
        self.sensitive_topics = config_data.get('sensitive_topics', {})
        
        # Recompile patterns
        self._compile_patterns()
        
        logger.info(f"Cultural adapter loaded from {file_path}")


def main():
    """Example usage of CulturalAdapter."""
    adapter = CulturalAdapter()
    
    # Test inputs
    test_inputs = [
        "তুমি ডাক্তারের কাছে যাও",
        "তোর মাথাব্যথা হচ্ছে কেন?",
        "আমি খুব দুঃখিত এবং আত্মহত্যার কথা ভাবছি",
        "আমার গর্ভাবস্থায় কি খাওয়া উচিত?",
        "বাচ্চার জ্বর হয়েছে কি করব?"
    ]
    
    print("Cultural Adaptation Examples:")
    print("=" * 50)
    
    for text in test_inputs:
        print(f"\nOriginal: {text}")
        
        # Adapt input
        adapted_input = adapter.adapt_input(text)
        print(f"Adapted Input: {adapted_input}")
        
        # Detect sensitive topic
        sensitive_topic = adapter.detect_sensitive_topic(text)
        if sensitive_topic:
            print(f"Sensitive Topic: {sensitive_topic}")
        
        # Adapt output (simulating model response)
        adapted_output = adapter.adapt_output(f"আপনার সমস্যার জন্য ডাক্তারের পরামর্শ নিন।", sensitive_topic)
        print(f"Adapted Output: {adapted_output}")
        
        # Validate appropriateness
        is_appropriate, issues = adapter.validate_cultural_appropriateness(adapted_output)
        print(f"Culturally Appropriate: {is_appropriate}")
        if issues:
            print(f"Issues: {issues}")
        
        print("-" * 30)


if __name__ == "__main__":
    main()
