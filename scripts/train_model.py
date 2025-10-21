#!/usr/bin/env python3
"""
Training script for Bengali Medical Chatbot.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
import pandas as pd
from transformers import (
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    EarlyStoppingCallback, get_linear_schedule_with_warmup
)
from datasets import Dataset
import wandb

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.bengali_medical_model import BengaliMedicalModel
from data.preprocessor import MedicalDataPreprocessor
from training.trainer import MedicalTrainer
from utils.logger import setup_logger

logger = setup_logger(__name__)


def load_config(config_path: str) -> Dict:
    """Load training configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def prepare_dataset(data_path: str, preprocessor: MedicalDataPreprocessor) -> Dict[str, Dataset]:
    """Prepare training datasets."""
    logger.info(f"Loading data from {data_path}")
    
    # Load processed data
    train_df = pd.read_csv(Path(data_path) / "train.csv")
    val_df = pd.read_csv(Path(data_path) / "validation.csv")
    test_df = pd.read_csv(Path(data_path) / "test.csv")
    
    logger.info(f"Loaded datasets - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    return {
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    }


def setup_training_arguments(config: Dict, output_dir: str) -> TrainingArguments:
    """Setup training arguments."""
    training_config = config['training']
    hyperparams = training_config['hyperparameters']
    
    return TrainingArguments(
        output_dir=output_dir,
        
        # Training hyperparameters
        num_train_epochs=hyperparams['num_epochs'],
        per_device_train_batch_size=hyperparams['batch_size'],
        per_device_eval_batch_size=hyperparams['batch_size'],
        gradient_accumulation_steps=hyperparams['gradient_accumulation_steps'],
        
        # Optimization
        learning_rate=hyperparams['learning_rate'],
        weight_decay=hyperparams['weight_decay'],
        warmup_ratio=hyperparams['warmup_ratio'],
        max_grad_norm=hyperparams['max_grad_norm'],
        
        # Evaluation and saving
        evaluation_strategy=training_config['evaluation']['evaluation_strategy'],
        eval_steps=training_config['evaluation']['eval_steps'],
        save_strategy=training_config['checkpointing']['save_strategy'],
        save_steps=training_config['checkpointing']['save_steps'],
        save_total_limit=training_config['checkpointing']['save_total_limit'],
        load_best_model_at_end=training_config['checkpointing']['load_best_model_at_end'],
        metric_for_best_model=training_config['checkpointing']['metric_for_best_model'],
        greater_is_better=training_config['checkpointing']['greater_is_better'],
        
        # Logging
        logging_strategy=training_config['logging']['logging_strategy'],
        logging_steps=training_config['logging']['logging_steps'],
        
        # Hardware optimization
        fp16=training_config['mixed_precision']['fp16'],
        bf16=training_config['mixed_precision']['bf16'],
        gradient_checkpointing=training_config['gradient_checkpointing'],
        dataloader_num_workers=training_config['dataloader']['num_workers'],
        dataloader_pin_memory=training_config['dataloader']['pin_memory'],
        
        # Reproducibility
        seed=training_config['seed'],
        
        # Reporting
        report_to=["wandb"] if training_config['logging']['wandb']['enabled'] else [],
        run_name=f"{training_config['experiment_name']}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
        
        # Additional settings
        remove_unused_columns=False,
        push_to_hub=False,
    )


def setup_wandb(config: Dict):
    """Setup Weights & Biases logging."""
    wandb_config = config['training']['logging']['wandb']
    
    if wandb_config['enabled']:
        wandb.init(
            project=wandb_config['project'],
            entity=wandb_config.get('entity'),
            tags=wandb_config.get('tags', []),
            config=config
        )
        logger.info("Weights & Biases initialized")


def tokenize_function(examples, tokenizer, max_input_length: int, max_target_length: int):
    """Tokenize examples for training."""
    # Tokenize inputs
    model_inputs = tokenizer(
        examples['input_text'],
        max_length=max_input_length,
        truncation=True,
        padding=False
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples['target_text'],
            max_length=max_target_length,
            truncation=True,
            padding=False
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Bengali Medical Chatbot")
    parser.add_argument(
        "--config",
        default="config/training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Directory containing processed training data"
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/models",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with reduced data"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="Path to checkpoint to resume training from"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_file = Path(args.output_dir) / "training.log"
    logger = setup_logger(__name__, log_file=str(log_file))
    
    logger.info("Starting Bengali Medical Chatbot training")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup Weights & Biases
    setup_wandb(config)
    
    # Initialize model
    logger.info("Initializing Bengali Medical Model...")
    model = BengaliMedicalModel(args.config.replace('training_config.yaml', 'model_config.yaml'))
    
    # Initialize preprocessor
    preprocessor = MedicalDataPreprocessor(args.config.replace('training_config.yaml', 'data_config.yaml'))
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    datasets = prepare_dataset(args.data_dir, preprocessor)
    
    # Debug mode - use smaller dataset
    if args.debug:
        logger.info("Debug mode: Using smaller dataset")
        datasets['train'] = datasets['train'].select(range(min(1000, len(datasets['train']))))
        datasets['validation'] = datasets['validation'].select(range(min(200, len(datasets['validation']))))
    
    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    max_input_length = config['model']['architecture']['max_input_length']
    max_target_length = config['model']['architecture']['max_output_length']
    
    tokenized_datasets = {}
    for split, dataset in datasets.items():
        tokenized_datasets[split] = dataset.map(
            lambda examples: tokenize_function(
                examples, 
                model.tokenizer, 
                max_input_length, 
                max_target_length
            ),
            batched=True,
            remove_columns=dataset.column_names
        )
    
    # Setup training arguments
    training_args = setup_training_arguments(config, args.output_dir)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=model.tokenizer,
        model=model.base_model,
        padding=True
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model.base_model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=model.tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=config['training']['early_stopping']['patience'],
            early_stopping_threshold=config['training']['early_stopping']['threshold']
        )] if config['training']['early_stopping']['enabled'] else []
    )
    
    # Training
    logger.info("Starting training...")
    
    try:
        if args.resume_from_checkpoint:
            logger.info(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            trainer.train()
        
        logger.info("Training completed successfully!")
        
        # Save final model
        logger.info("Saving final model...")
        final_model_path = Path(args.output_dir) / "final_model"
        model.save_model(str(final_model_path))
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
        
        logger.info("Test Results:")
        for key, value in test_results.items():
            logger.info(f"  {key}: {value}")
        
        # Save test results
        results_file = Path(args.output_dir) / "test_results.json"
        import json
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2)
        
        # Log to wandb
        if config['training']['logging']['wandb']['enabled']:
            wandb.log({"test_results": test_results})
            wandb.finish()
        
        logger.info(f"Training completed! Model saved to: {final_model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if config['training']['logging']['wandb']['enabled']:
            wandb.finish(exit_code=1)
        raise
    
    return 0


if __name__ == "__main__":
    exit(main())
