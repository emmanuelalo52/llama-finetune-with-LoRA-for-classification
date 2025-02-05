# LLaMa Text Classification Project

A PyTorch implementation of a modified LLaMa architecture for text classification tasks, featuring quantization, custom data processing, and efficient training pipelines.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Data Processing](#data-processing)
- [Training](#training)
- [Inference](#inference)
- [Contributing](#contributing)

## Overview

This project implements a modified version of the LLaMa architecture specifically designed for text classification tasks. It includes features like QLoRA quantization, custom data processing pipelines, and efficient training procedures. The implementation is based on the AG News dataset but can be adapted for other text classification tasks.

## Features

- Modified LLaMa architecture for classification tasks
- QLoRA quantization support
- Custom data preprocessing and cleaning
- Efficient attention mechanisms
- Rotary positional embeddings
- Multi-head attention with key-value caching
- Comprehensive text cleaning utilities
- Flexible dataset loading and processing
- Training with Hugging Face's Trainer API

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
datasets
huggingface_hub
peft
bitsandbytes
sklearn
pandas
emoji
nltk
```

## Project Structure

```
.
├── Llamaclassification.py    # Main model architecture
├── Dataloader.py            # Dataset loading and processing
├── inference.py             # Training and inference pipeline
├── data_cleaning.py         # Text preprocessing utilities
└── Dependencies.py          # Required package imports
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llama-classification.git
cd llama-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (for stopwords):
```python
import nltk
nltk.download('stopwords')
```

## Usage

### 1. Data Preparation

The project uses the AG News dataset by default. To use your own dataset, modify the `Dataloader.py` file:

```python
from datasets import load_dataset

# Load your dataset
dataset = load_dataset("your_dataset_name")

# Process your dataset similar to the AG News example
```

### 2. Text Cleaning

Use the `FeatureCleaner` class for text preprocessing:

```python
from data_cleaning import FeatureCleaner

cleaner = FeatureCleaner()
cleaned_text = cleaner("Your text here")
```

### 3. Training

Run the training pipeline:

```python
python inference.py
```

This will:
- Load and preprocess the dataset
- Initialize the model with QLoRA quantization
- Train the model using the Trainer API
- Evaluate the model on the test set

### 4. Inference

For inference on new text:

```python
from inference import classify_text, tokenizer

# Preprocess your text
tokens = tokenizer("Your text here", return_tensors="pt", padding=True, truncation=True)
prediction = classify_text(tokens['input_ids'], tokens['attention_mask'])
```

## Model Architecture

The model architecture consists of several key components:

1. **Embedding Layer**: Converts input tokens to embeddings
2. **Encoder Blocks**: Multiple layers of:
   - Self-attention mechanism
   - Feed-forward network
   - RMSNorm normalization
3. **Classification Head**: Final layer for class prediction

### Attention Mechanism

The attention mechanism includes:
- Multi-head attention with separate Q, K, V projections
- Key-value caching for efficient processing
- Rotary positional embeddings
- Grouped-query attention support

## Data Processing

The data processing pipeline includes:

1. **Text Cleaning**:
   - URL removal
   - Hashtag removal
   - Special character cleaning
   - Emoji handling
   - Slang expansion
   - Date normalization

2. **Tokenization**:
   - Using LLaMa tokenizer
   - Padding and truncation
   - Attention mask generation

3. **Label Processing**:
   - Label encoding
   - Label name mapping

## Training

The training process uses Hugging Face's Trainer API with the following features:

- QLoRA quantization for efficient training
- Customizable training arguments
- Evaluation during training
- Model checkpointing
- Logging and monitoring

Training parameters can be modified in `inference.py`:

```python
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
