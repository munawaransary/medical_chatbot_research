from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bengali-medical-chatbot",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A culturally-aware medical chatbot for Bengali healthcare",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/munawaransary/medical_chatbot_research",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "notebook>=7.0.0",
        ],
        "deployment": [
            "docker>=6.1.0",
            "boto3>=1.28.0",
            "azure-storage-blob>=12.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bengali-med-train=src.training.trainer:main",
            "bengali-med-evaluate=src.training.evaluator:main",
            "bengali-med-chat=src.inference.chatbot:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    keywords="bengali, medical, chatbot, nlp, healthcare, ai, machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/munawaransary/medical_chatbot_research/issues",
        "Source": "https://github.com/munawaransary/medical_chatbot_research",
        "Documentation": "https://github.com/munawaransary/medical_chatbot_research/docs",
    },
)
