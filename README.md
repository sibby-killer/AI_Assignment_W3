# 🧠 AI Assignment Week 3 – Machine Learning Web App
An interactive Streamlit-based AI application that integrates multiple machine learning models — including handwritten digit recognition, iris classification, and text analysis — into a single, user-friendly web interface.

https://img.shields.io/badge/Python-3.8%252B-blue
https://img.shields.io/badge/TensorFlow-2.13%252B-orange
https://img.shields.io/badge/Streamlit-1.28%252B-red
https://img.shields.io/badge/License-MIT-green

## 🧩 Project Overview
This project demonstrates practical machine learning and data science concepts through a visual and interactive web app built using TensorFlow, Scikit-learn, and Streamlit. It provides an easy-to-use interface for training, testing, and visualizing models on different datasets.

## ⚙️ Installation
1. Clone the Repository
bash
git clone https://github.com/christinemirimba/AI_Assignment_W3.git
cd AI_Assignment_W3
2. Create a Virtual Environment (Recommended)
bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
3. Install Dependencies
bash
pip install -r requirements.txt
4. Download spaCy English Model
bash
python -m spacy download en_core_web_sm
🚀 Usage
1. Train the Machine Learning Models
bash
python main.py
This will:

Train the iris classification model

Train the digit recognition CNN

Perform text analysis

Generate visualizations in the output/ directory

2. Launch the Interactive Web Application
bash
streamlit run app.py
The app will open in your default browser at http://localhost:8501

## 🧠 Machine Learning Models
1. Handwritten Digit Recognition
Architecture: Convolutional Neural Network (CNN)

Layers: 5 convolutional, 3 dense layers with dropout

Accuracy: >98% on MNIST test set

Features: Batch normalization, data augmentation

2. Iris Classification
Algorithm: Decision Tree Classifier

Features: Sepal length, sepal width, petal length, petal width

Classes: Setosa, Versicolor, Virginica

Accuracy: >95%

3. Text Analysis
Sentiment Analysis: Custom lexicon-based approach

NER: spaCy for entity recognition

Pattern Matching: Phrase matcher for product detection

## 🎯 Web Application Features
✏️ Drawing Interface
Interactive canvas for digit drawing

Real-time classification

Confidence scores and probability distributions

Download drawings as PNG files

📤 Image Upload
Supports PNG, JPG, JPEG

Automatic preprocessing and enhancement

Batch processing capabilities

🎲 Sample Testing
Pre-generated digit samples

Model accuracy verification

Performance benchmarking

📊 Output and Analysis
The system produces rich analytical outputs including:

Model performance metrics (accuracy, precision, recall)

Training history visualizations

Confusion matrices

Feature importance charts

Sentiment analysis reports

Entity recognition results

## 🛠️ Technical Details
Dependencies
txt
tensorflow>=2.13.0
streamlit>=1.28.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
numpy>=1.24.0
spacy>=3.7.0
streamlit-drawable-canvas>=0.9.0
Pillow>=10.0.0
## Model Performance
Model	Accuracy	Precision	Recall	Training Time
Digit Recognition	98.2%	98.1%	98.0%	~5 minutes
Iris Classification	96.7%	96.5%	96.7%	<1 second
Text Analysis	-	-	-	<1 second
🎮 How to Use the Web App
Drawing Tab ✏️
Draw a digit (0–9) in the canvas

Click "Classify Drawing"

View real-time predictions and confidence scores

Upload Tab 📤
Upload an image containing a handwritten digit

Click "Analyze Uploaded Image"

View classification results

Sample Tab 🎲
Generate sample digits

Test model accuracy

Download sample images

🔧 Customization
Adding New Models
Add model training code to main.py

Update the web interface in app.py

Add new visualization functions

Modifying Text Analysis
Update product reviews in the text_analysis() function

Modify sentiment lexicons for different domains

Add new entity recognition patterns

Styling Changes
Modify CSS in the st.markdown() sections

Update color schemes and layouts

Add custom Streamlit components

## 🐛 Troubleshooting
Common Issues
TensorFlow not installing

bash
pip install tensorflow-cpu  # For CPU-only systems
spaCy model not found

bash
python -m spacy download en_core_web_sm
Streamlit canvas not working

bash
pip install streamlit-drawable-canvas
Out of memory during training

Reduce batch size in main.py

Use a smaller model architecture

Enable GPU memory growth

Performance Tips
Use GPU for faster training (tensorflow-gpu)

Reduce image size for quicker processing

Use smaller batches if memory is limited

Enable early stopping to prevent overfitting

## 📈 Results and Demos
🧮 Sample Predictions
Real-time digit classification with confidence scores

📊 Model Analytics
Model accuracy and loss during training

💬 Text Analysis
Product review sentiment distribution

## 🤝 Contributing
We welcome contributions!

Fork the repository

Create a feature branch

bash
git checkout -b feature/amazing-feature
Commit your changes

bash
git commit -m "Add amazing feature"
Push to the branch

bash
git push origin feature/amazing-feature
Open a Pull Request

## 📝 License
This project is licensed under the MIT License — see the LICENSE file for details.

## 👥 Team
Name	Role	Email
Christine Mirimba	Machine Learning Engineer	mirimbachristine@gmail.com
Alfred Nyongesa	Data Analyst & System Optimization	alfred.dev8@gmail.com
Hannah Shekinah	AI Ethics & Sustainability Specialist	hannahshekinah@gmail.com
Joelina Quarshie	Technical Writer & Research Coordinator	joelinakq@gmail.com

GitHub: @christinemirimba

Project: Digit Recognition System
## 🙏 Acknowledgments
MNIST dataset providers

Scikit-learn team for machine learning tools

Streamlit team for the amazing web framework

TensorFlow team for deep learning capabilities

## ⭐ If you find this project helpful, don't forget to give it a star!

## Built with ❤️ using Python, TensorFlow, and Streamlit

