# Fake News Detection using DistilBERT

## Overview

This project implements a **Fake News Detection System** using the **DistilBERT transformer model** and transfer learning techniques. The model is fine-tuned on the Fake.csv and True.csv datasets to classify news articles as either **Fake** or **Real**.

Unlike traditional machine learning and recurrent neural network approaches, transformer-based models capture contextual relationships within text more effectively, resulting in significantly improved performance.

The project was developed using **PyTorch** and the **Hugging Face Transformers** library and includes extensive evaluation using multiple classification metrics and visualization techniques.

---

## Features

* Transformer-based text classification using DistilBERT
* Transfer learning with pre-trained language models
* Binary classification of fake and real news articles
* Prevention of data leakage and duplicate samples
* Dynamic tokenization and padding
* Performance evaluation using multiple metrics
* Confusion matrix visualization
* ROC-AUC analysis
* Model saving and inference support

---

## Dataset

The project uses two publicly available datasets:

* **True.csv** – Contains authentic news articles.
* **Fake.csv** – Contains fabricated news articles.

### Dataset Composition

| Class | Description |
| ----- | ----------- |
| 0     | Fake News   |
| 1     | Real News   |

### Preprocessing

To ensure reliable evaluation:

* Combined title and article text.
* Removed the subject column to avoid information leakage.
* Removed duplicate samples.
* Ensured no overlap between training and testing data.
* Used DistilBERT tokenizer for text encoding.
* Applied truncation and dynamic padding.

---

## Project Structure

```text
Fake-News-Detection-Using-DistilBERT/
│
├── dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── Fake_News_Detection.ipynb
│ 
│
├── Images/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── classification_report.txt
│   └── results.csv
│
├── requirements.txt
├── README.md
```

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* DistilBERT
* Datasets
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn

---

## Model Architecture

### DistilBERT

DistilBERT is a lightweight and efficient version of BERT that retains most of BERT's language understanding capabilities while reducing computational requirements.

Model used:

```python
distilbert-base-uncased
```

### Training Configuration

| Parameter               |   Value |
| ----------------------- | ------: |
| Learning Rate           |    2e-5 |
| Epochs                  |       3 |
| Batch Size              |      16 |
| Maximum Sequence Length |     256 |
| Weight Decay            |    0.01 |
| Optimizer               |   AdamW |
| Framework               | PyTorch |

---

## Performance Metrics

### Evaluation Results

| Metric        |  Score |
| ------------- | -----: |
| Accuracy      | 99.91% |
| Precision     | 99.98% |
| Recall        | 99.86% |
| F1 Score      | 99.92% |
| ROC-AUC Score | 99.92% |

---

## Classification Report

```text
              precision    recall  f1-score   support

Fake News        1.00       1.00      1.00      3582
Real News        1.00       1.00      1.00      4239

accuracy                              1.00      7821
macro avg         1.00       1.00      1.00      7821
weighted avg      1.00       1.00      1.00      7821
```

---

## Confusion Matrix

| Actual Class | Predicted Fake | Predicted Real |
| ------------ | -------------: | -------------: |
| Fake News    |           3581 |              1 |
| Real News    |              6 |           4233 |

Only **7 misclassifications** were observed out of **7,821 test samples**.

---

## Training Results

| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall | F1 Score |
| ----- | ------------: | --------------: | -------: | --------: | -----: | -------: |
| 1     |        0.0252 |          0.0093 |   99.78% |    99.65% | 99.95% |   99.80% |
| 2     |        0.0029 |          0.0077 |   99.91% |    99.98% | 99.86% |   99.92% |
| 3     |        0.0000 |          0.0092 |   99.87% |    99.86% | 99.91% |   99.88% |

---

## Model Training Pipeline

1. Load Fake.csv and True.csv datasets.
2. Merge and label the classes.
3. Remove duplicate records.
4. Split into training and testing sets.
5. Tokenize text using DistilBERT tokenizer.
6. Fine-tune DistilBERT for binary classification.
7. Evaluate using classification metrics.
8. Generate confusion matrix and ROC-AUC score.
9. Save the trained model for future inference.

---

## Saving the Model

```python
model.save_pretrained("distilbert_model")
tokenizer.save_pretrained("distilbert_model")
```

---

## Future Improvements

* Compare DistilBERT with RoBERTa and DeBERTa.
* Hyperparameter tuning.
* Deploy the model using Streamlit or Flask.
* Build a real-time fake news detection web application.
* Support multilingual news classification.
* Integrate explainable AI techniques for prediction interpretation.

---

## Conclusion

This project demonstrates the effectiveness of transformer-based transfer learning for fake news classification. By leveraging DistilBERT and carefully preventing data leakage, the model achieved excellent performance with an accuracy of 99.91% and an F1-score of 99.92%, significantly outperforming traditional neural network approaches.
