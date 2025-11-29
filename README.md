# Multi-Label Legal Document Classification (ECHR)

## Overview
This project implements a multi-label text classification system for predicting violated articles
of the European Convention on Human Rights (ECHR) from legal case descriptions. 
The dataset is taken from the **LEXGLUE – ECTHR-B** benchmark via HuggingFace.

The solution is fully implemented and tested in **Google Colab** using **PyTorch** and 
**HuggingFace Transformers**, with **Legal-BERT** as the main backbone model.

---

## Dataset
**Dataset:** lex_glue / ecthr_b  
**Source:** HuggingFace Datasets

Each sample:
- `text` — factual paragraph from the case
- `labels` — list of violated ECHR articles (multi-label)

Dataset splits:
- Train ~9000 samples  
- Validation ~1000 samples  
- Test ~1000 samples  

---

## Project Pipeline

1. **Dataset Loading**
   - Load ECTHR-B from HuggingFace
   - Convert to pandas for inspection

2. **Exploratory Data Analysis (EDA)**
   - Text-length distribution
   - Label frequency analysis
   - Co-occurrence heatmaps
   - Labels-per-document statistics

3. **Preprocessing**
   - Text cleaning (lowercase, punctuation removal, whitespace normalization)
   - Handling missing/duplicate entries
   - Tokenization using Legal-BERT tokenizer

4. **Label Encoding**
   - Multi-hot encoding using `MultiLabelBinarizer`
   - Conversion of labels to fixed-size float32 tensors

5. **Model Setup**
   - Baseline: Pretrained **Legal-BERT** without fine-tuning
   - Final Model: Same Legal-BERT **fine-tuned on ECHR dataset**

6. **Training**
   - Optimizer: AdamW
   - Loss: Binary Cross Entropy with Logits (BCEWithLogits)
   - Learning Rate: `2e-5`
   - Batch Size: `8`
   - Epochs: `3`

7. **Evaluation**
   - Metrics:
     - Micro F1-score
     - Macro F1-score
     - Precision
     - Recall

8. **Prediction**
   - Generate predictions on test set
   - Decode multi-label outputs
   - Store results in dataframe (`test_df`) with predicted labels

---

## Model Architecture

- **Base Model:** `nlpaueb/legal-bert-base-uncased`
- **Classifier Head:** Multi-label sigmoid layer
- **Activation Function:** Sigmoid
- **Loss:** BCEWithLogitsLoss

---

## Results

| Model              | Micro-F1 | Macro-F1 |
|--------------------|-----------|-----------|
| Baseline (No FT)  | ~0.52     | ~0.41     |
| Fine-Tuned Model | ~0.72     | ~0.63     |

Fine-tuning significantly improves predictive performance for domain-specific legal data.

---

## Challenges & Solutions

| Challenge | Solution |
|----------|-------------|
| Variable-length labels | Multi-hot encoding using `MultiLabelBinarizer` |
| Data type mismatch | Converted labels to `float32` for BCE loss |
| Label imbalance | Handled with suitable thresholding and model training |
| Long documents | Fixed max token length (512) with truncation |
| Dataset corruption | Re-order pipeline: encode labels BEFORE tokenizing |

---

## Technologies Used

- Python 3
- Google Colab
- HuggingFace Transformers
- PyTorch
- scikit-learn
- pandas
- matplotlib
- seaborn

---

## How to Run (Google Colab)

1. Upload notebook to Colab or create a new notebook.
2. Install dependencies:
   pip install datasets transformers sklearn evaluate accelerate
3. Run cells sequentially:
   - Data loading
   - EDA
   - Preprocessing
   - Training
   - Evaluation
4. Download `test_predictions.csv` for analysis.

---

## Output Files

- `ecthr_full_dataset.csv` – complete dataset
- `test_predictions.csv` – predictions on test set
- Saved fine-tuned model (HuggingFace output folder)

---

## Author / Course Context

This work is part of an NLP assignment on Multi-Label Classification using legal documents 
from the ECHR. The focus is on building an end-to-end machine learning pipeline and evaluating 
the impact of domain-specific fine-tuning.