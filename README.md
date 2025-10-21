# BengaliMed: A Culturally-Aware Medical Chatbot for Bengali Healthcare

## Overview
This repository contains the implementation of BengaliMed, a research-focused Bengali medical chatbot designed to provide culturally-appropriate healthcare assistance to Bengali speakers. This project aims to bridge the gap in medical AI resources for the 230M+ Bengali-speaking population worldwide.

## 🎯 Research Objectives
- Create the first large-scale Bengali medical conversational dataset
- Develop a cultural adaptation framework for medical chatbots
- Establish evaluation methodologies for cross-lingual medical AI
- Provide an open-source Bengali medical chatbot system

## 🏗️ Project Structure
```
medical_chatbot/
├── README.md                    # Project overview and setup
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── config/                      # Configuration files
│   ├── model_config.yaml       # Model hyperparameters
│   ├── data_config.yaml        # Data processing settings
│   └── training_config.yaml    # Training configurations
├── data/                        # Dataset storage
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned and processed data
│   ├── translated/             # Bengali translations
│   └── augmented/              # Augmented datasets
├── src/                         # Source code
│   ├── __init__.py
│   ├── data/                   # Data processing modules
│   │   ├── __init__.py
│   │   ├── translator.py       # Translation pipeline
│   │   ├── preprocessor.py     # Data cleaning and preprocessing
│   │   ├── augmentor.py        # Data augmentation
│   │   └── validator.py        # Quality validation
│   ├── models/                 # Model implementations
│   │   ├── __init__.py
│   │   ├── bengali_medical_model.py  # Main model class
│   │   ├── cultural_adapter.py       # Cultural adaptation module
│   │   └── fine_tuner.py            # Fine-tuning utilities
│   ├── training/               # Training scripts
│   │   ├── __init__.py
│   │   ├── trainer.py          # Main training loop
│   │   ├── evaluator.py        # Evaluation metrics
│   │   └── callbacks.py        # Training callbacks
│   ├── inference/              # Inference and deployment
│   │   ├── __init__.py
│   │   ├── chatbot.py          # Main chatbot interface
│   │   └── api.py              # REST API endpoints
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── logger.py           # Logging utilities
│       ├── metrics.py          # Evaluation metrics
│       └── helpers.py          # General helper functions
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb     # Dataset analysis
│   ├── 02_translation_pipeline.ipynb # Translation experiments
│   ├── 03_model_training.ipynb       # Training experiments
│   └── 04_evaluation.ipynb          # Results analysis
├── scripts/                     # Utility scripts
│   ├── download_data.py        # Data download automation
│   ├── train_model.py          # Training script
│   ├── evaluate_model.py       # Evaluation script
│   └── deploy_model.py         # Deployment script
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── test_data/              # Data processing tests
│   ├── test_models/            # Model tests
│   └── test_utils/             # Utility tests
├── docs/                        # Documentation
│   ├── research_paper_outline.md    # Paper structure
│   ├── dataset_strategy.md          # Data creation strategy
│   ├── project_roadmap.md           # Development roadmap
│   └── api_documentation.md         # API docs
├── experiments/                 # Experiment tracking
│   ├── logs/                   # Training logs
│   ├── models/                 # Saved model checkpoints
│   └── results/                # Experiment results
└── deployment/                  # Deployment configurations
    ├── docker/                 # Docker configurations
    ├── streamlit_app.py        # Demo application
    └── requirements_deploy.txt  # Deployment dependencies
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/munawaransary/medical_chatbot_research.git
cd medical_chatbot_research

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
```bash
# Download datasets
python scripts/download_data.py

# Process and translate data
python scripts/process_data.py --config config/data_config.yaml
```

### 3. Model Training
```bash
# Train the model
python scripts/train_model.py --config config/training_config.yaml
```

### 4. Run Demo
```bash
# Start the Streamlit demo
streamlit run deployment/streamlit_app.py
```

## 📊 Datasets
- **Primary**: Kaggle AI Medical Chatbot datasets (English)
- **Secondary**: BanglaHealth paraphrase dataset (200K Bengali health sentences)
- **Augmentation**: Bengali medical textbooks and admission materials
- **Target Size**: 250K Bengali medical interactions

## 🔬 Research Contributions
1. **First large-scale Bengali medical conversational dataset**
2. **Cultural adaptation framework for medical chatbots**
3. **Cross-lingual medical knowledge transfer methodology**
4. **Comprehensive evaluation framework for Bengali medical AI**

## 📈 Performance Targets
- **Medical Accuracy**: >90% expert validation
- **Cultural Appropriateness**: >95% community approval
- **Linguistic Quality**: >85% native speaker satisfaction
- **Response Relevance**: >88% user satisfaction

## 🏥 Specialized Domains
- **Mental Health**: Depression, anxiety, cultural sensitivity
- **Pediatric Medicine**: Child health, vaccination, development
- **Women's Health**: Reproductive health, pregnancy care
- **General Medicine**: Common symptoms, preventive care

## 🛠️ Technology Stack
- **Framework**: PyTorch, Transformers (Hugging Face)
- **Models**: BanglaT5, mBERT, Custom Bengali embeddings
- **Translation**: Google Translate API + Manual correction
- **Database**: PostgreSQL for medical knowledge
- **Frontend**: Streamlit for demo, FastAPI for production
- **Deployment**: Docker containers, cloud-ready

## 📝 Research Timeline (All these are tentative)
- **Phase 1** (Weeks 1-3): Literature review and dataset analysis
- **Phase 2** (Weeks 4-8): Data preparation and translation
- **Phase 3** (Weeks 9-16): Model development and training
- **Phase 4** (Weeks 17-20): Evaluation and validation
- **Phase 5** (Weeks 21-24): Research publication

<!-- ## 🎯 Target Publications
- **Primary**: ACL 2025, EMNLP 2025
- **Secondary**: AMIA 2025, ICON 2025
- **Journal**: Journal of Medical Internet Research -->

## ⚠️ Important Disclaimers
- This system is for research purposes only
- Not intended to replace professional medical advice
- Always consult healthcare professionals for medical decisions
- Cultural adaptations are based on general patterns and may not apply to all individuals

## 🤝 Contributing
We welcome contributions from the research community. Please see our contribution guidelines and code of conduct.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact
- **Researcher**: Mim Raihan, Munawar Mahtab Ansary
- **Institution**: BRACU
- **Email**: munawar.mahtab.ansary@g.bracu.ac.bd
- **Project Repository**: https://github.com/munawaransary/medical_chatbot_research

## 🙏 Acknowledgments
- BanglaHealth dataset creators
- Kaggle medical dataset contributors
- Bengali medical experts and linguists
- Open-source NLP community

---
*This project aims to democratize healthcare access for Bengali speakers through culturally-aware AI technology.*
