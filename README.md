# Fake News Detection Project

**Author:** Martha Kachweka  
**Task:** TBI Technical Test  
**Date:** June 2025

## Project Overview

This project investigates binary classification techniques for distinguishing fake and real news using a text-based dataset spanning three domains: News, Politics, and Other. The goal is to build an effective machine learning model capable of detecting and classifying fake news articles with high accuracy.

## Abstract

This project aims to build a machine learning model to detect and classify fake news articles using labeled text samples. The approach combines baseline models with advanced ensemble techniques and feature engineering to achieve optimal performance.

### Key Features:
- **Baseline Models**: Logistic Regression and Naive Bayes for establishing reference performance
- **Advanced Models**: Random Forest, XGBoost, and LightGBM ensemble methods
- **Feature Engineering**: Text length analysis, word counting, punctuation analysis, and bias keyword detection
- **High Performance**: Achieved over 99% accuracy with XGBoost and LightGBM models

## Dataset

The project uses multiple datasets containing both fake and real news articles:

- `Fake.zip` - Collection of fake news articles
- `True.zip` - Collection of real news articles  
- `PolitiFact_fake_news_content.zip` - PolitiFact fake news dataset
- `PolitiFact_real_news_content.zip` - PolitiFact real news dataset

## Project Structure

```
TBI/
├── README.md                           # Project documentation
├── TBI_TechTest.ipynb                 # Main Jupyter notebook with analysis
├── Fake.zip                           # Fake news dataset
├── True.zip                           # Real news dataset
├── PolitiFact_fake_news_content.zip   # PolitiFact fake news
└── PolitiFact_real_news_content.zip   # PolitiFact real news
```

## Dependencies

The project requires the following Python libraries:

```python
# Data manipulation and analysis
numpy
pandas

# Visualization
matplotlib
seaborn
wordcloud

# Natural Language Processing
textblob
scikit-learn

# Machine Learning Models
lightgbm
xgboost

# Additional utilities
scipy
warnings
```

## Installation

1. Clone or download this repository
2. Install required dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn wordcloud textblob scikit-learn lightgbm xgboost scipy
   ```
3. Extract the dataset zip files
4. Open and run `TBI_TechTest.ipynb`

## Methodology

### 1. Data Preprocessing
- Text cleaning and normalization
- Feature extraction from news articles
- Dataset merging and preparation

### 2. Feature Engineering
- **Text Features**: Article length, word count
- **Linguistic Features**: Punctuation analysis (exclamation marks, question marks)
- **Content Analysis**: Bias-related keyword detection
- **TF-IDF Vectorization**: Converting text to numerical features

### 3. Model Development

#### Baseline Models
- **Logistic Regression**: Simple linear classifier
- **Naive Bayes**: Probabilistic text classifier

#### Advanced Ensemble Models
- **Random Forest**: Multiple decision tree ensemble
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Efficient gradient boosting

### 4. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report

## Results

The project demonstrates significant performance improvements with ensemble methods:

- **Baseline Models**: Reasonable performance establishing benchmarks
- **Ensemble Models**: Superior performance with XGBoost and LightGBM achieving **over 99% accuracy**
- **Feature Engineering**: Meaningful contribution to model performance through additional text-based features

## Key Insights

1. **Ensemble Methods Superiority**: XGBoost and LightGBM significantly outperformed baseline models
2. **Feature Engineering Impact**: Additional features beyond raw text improved classification accuracy
3. **Model Generalization**: High accuracy scores indicate strong model generalization capabilities
4. **Effective Misinformation Detection**: The combination of thoughtful feature design and advanced modeling proves effective for fake news detection

## Usage

1. Open `TBI_TechTest.ipynb` in Jupyter Notebook or JupyterLab
2. Run cells sequentially to:
   - Load and preprocess data
   - Engineer features
   - Train models
   - Evaluate performance
   - Compare results

## Future Improvements

- Implement cross-validation for more robust evaluation
- Explore deep learning approaches (LSTM, BERT)
- Add more sophisticated NLP features
- Experiment with ensemble combinations
- Deploy model as a web service

## License

This project is part of a technical assessment for TBI.

## Contact

For questions or discussions about this project, please reach out to Martha Kachweka.

---

*This project demonstrates the application of machine learning techniques for combating misinformation through automated fake news detection.*
