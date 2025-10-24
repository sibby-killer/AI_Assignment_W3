# 🌐 Digital Vision AI
A Complete Machine Learning & Deep Learning Application

🔗 **[Live Demo](https://digit-predict-ai.streamlit.app/)**

This project integrates Classical Machine Learning, Deep Learning, and Natural Language Processing into one interactive platform powered by Streamlit. Each task demonstrates an essential aspect of applied AI — classification, computer vision, and text understanding — deployed together with a clean tabbed navigation UI.

## 🧠 Project Overview

| **Task** | **Domain** | **Description** | **Technology** |
|-----------|-------------|-----------------|----------------|
| **Task 1** | Classical ML | *Iris Flower Species Classification* using **Decision Tree** | Scikit-learn |
| **Task 2** | Deep Learning | *Handwritten Digit Recognition* using **CNN** | TensorFlow / Keras |
| **Task 3** | NLP | *Named Entity Recognition and Sentiment Analysis* | spaCy |

---

## 🚀 Features

- 📊 **Iris Classification**  
  Predict species of iris flowers using a **Decision Tree Classifier** with real-time input and comprehensive dataset analysis.

- 🔢 **Digit Recognition**  
  Classify handwritten digits (**0–9**) using a **CNN model** achieving *over 98% accuracy*, featuring an interactive drawing interface and image upload option.

- 💬 **Text Analysis**  
  Extract entities (e.g., *products, brands*) and detect sentiment from user reviews using **spaCy NER** and *rule-based sentiment analysis*.

## 🧩 Unified Streamlit Interface 

- **Digit Recognition** - Interactive drawing and image upload
- **Iris Classification** - Real-time species prediction and dataset analysis  
- **Text Analysis** - Sentiment detection and entity extraction
- 📈 Interactive charts, confusion matrices, performance summaries, and comprehensive visualizations

## 📁 Project Structure
- AI_Assignment_W3/
- ├── app.py # Streamlit main application with 3-tab navigation
- ├── main.py # Complete ML pipeline training and evaluation
- ├── requirements.txt # Dependencies
- ├── models/ # Saved trained models
- │ ├── digit_recognition_model.h5
- │ └── iris_model.pkl
- ├── output/ # Visualizations and analysis results
- │ ├── iris_analysis/
- │ ├── digit_recognition/
- │ └── nlp_analysis/

---
## 🛠️ Installation

1. Using `venv` – ✅ *Recommended* **

```bash
# Clone the repository
git clone https://github.com/christinemirimba/AI_Assignment_W3.git
cd AI_Assignment_W3

# Create virtual environment
python -m venv ai_assignment_w3_env

# Activate environment
# Windows:
ai_assignment_w3_env\Scripts\activate
# Mac/Linux:
source ai_assignment_w3_env/bin/activate

# Install dependencies
pip install -r requirements.txt
 
# Download spaCy English model
python -m spacy download en_core_web_sm

# Install additional components
pip install streamlit-drawable-canvas joblib

# Run training pipeline
python main.py

# Launch the application
streamlit run app.py

# Deactivate environment when done
deactivate
2. Using Conda
bash
# Create conda environment
conda create -n ai_assignment_w3 python=3.9
conda activate ai_assignment_w3

# Install packages
pip install tensorflow scikit-learn spacy matplotlib seaborn pandas numpy streamlit pillow streamlit-drawable-canvas joblib

# Download spaCy model
python -m spacy download en_core_web_sm

# Run training and launch app
python main.py
streamlit run app.py

# Deactivate environment
conda deactivate
3. Quick Setup with pipreqs
bash
# Generate requirements from actual code usage
pip install pipreqs
pipreqs . --encoding utf-8 --force

# Install generated requirements
pip install -r requirements.txt

# Optional: CPU-only TensorFlow (if GPU not available)
pip install tensorflow-cpu

---

## 🚀 Usage

1. **Train All Machine Learning Models**

```bash
python main.py
This comprehensive training pipeline will:

Iris Classification: Train Decision Tree classifier and save model to models/iris_model.pkl

Digit Recognition: Train CNN model with data augmentation and save to models/digit_recognition_model.h5

Text Analysis: Perform NLP analysis on product reviews

Generate Visualizations: Create comprehensive analysis in output/ directory including:

Feature importance charts

Confusion matrices

Training history plots

Sentiment analysis results

Entity recognition visualizations

Launch Interactive Web Application

bash
streamlit run app.py
The application will open in your default browser at http://localhost:8501
---

## 🧠 Machine Learning Models
1. 🧩 Handwritten Digit Recognition
Architecture: Enhanced CNN with 5 convolutional and 3 dense layers

Features: Batch normalization, dropout, data augmentation

Accuracy: >98% on MNIST test set

Input: 28×28 grayscale images

Output: Digit classification (0–9)

2. 🌸 Iris Classification
Algorithm: Decision Tree Classifier (max_depth=3)

Features: Sepal length, sepal width, petal length, petal width

Classes: Setosa, Versicolor, Virginica

Accuracy: >95%

Real-time Prediction: Input measurements for instant species classification

3. 💬 Text Analysis
Sentiment Analysis: Rule-based approach with custom lexicons

Named Entity Recognition: spaCy for extracting products and brands

Pattern Matching: Phrase matcher for product detection

Visualization: Sentiment distribution and entity type analysis

🎯 Web Application Features
1. ✏️ Digit Recognition Tab
Interactive Canvas: Draw digits with real-time classification

Image Upload: Supports PNG, JPG, and JPEG formats

Sample Testing: Includes pre-generated digits for model verification

Confidence Scoring: Displays probability distributions and confidence levels

Download Functionality: Save your drawings as PNG files

2. 🌺 Iris Classification Tab
Real-time Prediction: Input flower measurements for instant species classification

Dataset Analysis: Explore comprehensive visualizations of the Iris dataset

Feature Importance: View visual representations of key features

Performance Metrics: Displays model accuracy, precision, and recall

Interactive Charts: Analyze feature distributions and correlation heatmaps

3. 📝 Text Analysis Tab
Sample Reviews: Includes pre-loaded product reviews for demonstration

Custom Text Input: Analyze your own text or reviews

Sentiment Detection: Classify text as Positive, Negative, or Neutral with confidence scores

Entity Extraction: Identify product names, brands, and organizations using NLP

Visual Analytics: View sentiment distributions and entity type charts

---

## 📊 Model Performance

| Model               | Accuracy | Precision | Recall | Training Time | Key Features                        |
|---------------------|----------|-----------|---------|---------------|-------------------------------------|
| **Digit Recognition** | 98.2%    | 98.1%     | 98.0%   | ~5 minutes    | CNN with data augmentation          |
| **Iris Classification** | 96.7%  | 96.5%     | 96.7%   | <1 second     | Decision Tree visualization         |
| **Text Analysis**     | -        | -         | -       | <1 second     | spaCy NER + rule-based sentiment    |
---

## 👥 Team

| Name               | Role                                      | Email                           |
|--------------------|-------------------------------------------|----------------------------------|
| **Christine Mirimba** | Machine Learning Engineer                 | mirimbachristine@gmail.com       |
| **Alfred Nyongesa**   | Data Analyst & System Optimization        | alfred.dev8@gmail.com            |
| **Hannah Shekinah**   | AI Ethics & Sustainability Specialist     | hannahshekinah@gmail.com         |
| **Joelina Quarshie**  | Technical Writer & Research Coordinator   | joelinakq@gmail.com              |

---

## 🙏 Acknowledgments

- MNIST dataset providers for digit recognition  
- Scikit-learn team for machine learning tools  
- Streamlit team for the amazing web framework  
- TensorFlow team for deep learning capabilities  
- spaCy team for natural language processing tools

---

## ⭐ Support
If you find this project helpful, please give it a star! Your support helps us continue improving and maintaining this application.

---

**Built with ❤️ using Python, TensorFlow, Scikit-learn, spaCy, and Streamlit**
