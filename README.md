# 🧠 AI Assignment Week 3 – Machine Learning Web App
An interactive Streamlit-based AI application that integrates multiple machine learning models — including handwritten digit recognition, iris classification, and text analysis — into a single, user-friendly web interface.

## 🔗 Live Demo: https://digit-predict-ai.streamlit.app/

## 🧩 Project Overview
This project demonstrates practical machine learning and data science concepts through a visual and interactive web app built using TensorFlow, Scikit-learn, and Streamlit. It provides an intuitive interface for training, testing, and visualizing models across diverse datasets.

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
## 🚀 Quick Setup Guide
Method 1: Using venv – ✅ Recommended
bash
python -m venv ai_assignment_w3_env
# Activate
ai_assignment_w3_env\Scripts\activate  # Windows
source ai_assignment_w3_env/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Optional: CPU-only TensorFlow
pip install tensorflow-cpu

# Download spaCy model
python -m spacy download en_core_web_sm

# Install Streamlit canvas
pip install streamlit-drawable-canvas

# Run training
python main.py

# Launch app
streamlit run app.py

# Deactivate
deactivate
Method 2: Using conda
bash
conda create -n ai_assignment_w3 python=3.9
conda activate ai_assignment_w3

pip install tensorflow scikit-learn spacy matplotlib seaborn pandas numpy streamlit pillow
python -m spacy download en_core_web_sm

python main.py
streamlit run app.py

conda deactivate
## 🚀 Usage
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

🧠 Machine Learning Models
1. Handwritten Digit Recognition
Architecture: CNN

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

## 📊 Output and Analysis
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
## 🎮 How to Use the Web App
Drawing Tab ✏️
Draw a digit (0–9) in the canvas

Click Classify Drawing

View predictions and confidence scores

Upload Tab 📤
Upload a digit image

Click Analyze Uploaded Image

View classification results

Sample Tab 🎲
Generate sample digits

Test model accuracy

Download sample images

🔧 Customization
Adding New Models
Add training code to main.py

Update UI in app.py

Add visualization functions

Modifying Text Analysis
Update reviews in text_analysis()

Modify sentiment lexicons

Add new entity patterns

Styling Changes
Modify CSS in st.markdown()

Update color schemes and layout

Add custom Streamlit components

## 🐛 Troubleshooting
Common Issues
TensorFlow not installing

bash
pip install tensorflow-cpu
spaCy model not found

bash
python -m spacy download en_core_web_sm
Streamlit canvas not working

bash
pip install streamlit-drawable-canvas
Out of memory during training

Reduce batch size

Use smaller architecture

Enable GPU memory growth

Performance Tips
Use GPU (tensorflow-gpu)

Reduce image size

Use smaller batches

Enable early stopping

### 📈 Results and Demos
🧮 Real-time digit classification

📊 Model accuracy and loss graphs

💬 Sentiment distribution from product reviews

## 🤝 Contributing
We welcome contributions!

bash
# Fork the repository
git checkout -b feature/amazing-feature

# Commit your changes
git commit -m "Add amazing feature"

# Push to the branch
git push origin feature/amazing-feature
Then, open a Pull Request 🚀

## 📝 License
This project is licensed under the MIT License — see the LICENSE file for details.

## 👥 Team
Name	Role	Email
Christine Mirimba	Machine Learning Engineer	mirimbachristine@gmail.com
Alfred Nyongesa	Data Analyst & System Optimization	alfred.dev8@gmail.com
Hannah Shekinah	AI Ethics & Sustainability Specialist	hannahshekinah@gmail.com
Joelina Quarshie	Technical Writer & Research Coordinator	joelinakq@gmail.com

## 👩‍💻 Author
Christine Mirimba Machine Learning Engineer • Full-stack Developer • UX Designer

📫 Email: mirimbachristine@gmail.com

🐙 GitHub: @christinemirimba

🎓 Project: Digit Recognition System

Passionate about building inclusive, data-driven solutions that empower communities and drive real-world change.

## 🙏 Acknowledgments
MNIST dataset providers

Scikit-learn team

Streamlit team

TensorFlow team

## ⭐ If you find this project helpful, don't forget to give it a star! Built with ❤️ using Python, TensorFlow, and Streamlit 