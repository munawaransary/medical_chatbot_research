"""
Bengali Medical Chatbot - A culturally-aware medical chatbot for Bengali healthcare.

This package provides tools and models for creating a research-focused Bengali medical
chatbot with cultural adaptation capabilities.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .models.bengali_medical_model import BengaliMedicalModel
from .inference.chatbot import BengaliMedicalChatbot
from .data.translator import MedicalTranslator
from .data.preprocessor import MedicalDataPreprocessor

__all__ = [
    "BengaliMedicalModel",
    "BengaliMedicalChatbot", 
    "MedicalTranslator",
    "MedicalDataPreprocessor",
]
