"""
Digital Vision AI - Interactive Web Application
Complete ML Pipeline with Three Modules
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import os
import io
import base64
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

# Page configuration
st.set_page_config(
    page_title="Digital Vision AI",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
        margin: 2px 0;
    }
    .tab-content {
        padding: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MODEL LOADING FUNCTIONS
# =============================================================================

@st.cache_resource
def load_digit_model():
    """Load the pre-trained digit recognition model"""
    try:
        import tensorflow as tf
        model_path = 'models/digit_recognition_model.h5'
        
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            return model
        else:
            st.warning("Digit recognition model not found. Please run main.py first.")
            return None
    except ImportError:
        st.error("TensorFlow not available. Please install TensorFlow.")
        return None
    except Exception as e:
        st.error(f"Error loading digit model: {e}")
        return None

@st.cache_resource
def load_iris_model():
    """Train or load Iris classification model"""
    try:
        # Load Iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Train model (or load if saved)
        model_path = 'models/iris_model.pkl'
        if os.path.exists(model_path):
            import joblib
            model = joblib.load(model_path)
        else:
            # Train new model
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            model = DecisionTreeClassifier(max_depth=3, random_state=42)
            model.fit(X_train, y_train)
            
            # Save model
            os.makedirs('models', exist_ok=True)
            joblib.dump(model, model_path)
        
        return model, iris
    except Exception as e:
        st.error(f"Error with Iris model: {e}")
        return None, None

def initialize_nlp():
    """Initialize NLP components"""
    try:
        import spacy
        from spacy.matcher import PhraseMatcher
        
        # Try to load spaCy model
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            st.warning("spaCy English model not found. Using basic text analysis.")
            nlp = None
        
        return nlp
    except ImportError:
        st.warning("spaCy not available. Using basic text analysis.")
        return None

# =============================================================================
# DIGIT RECOGNITION FUNCTIONS
# =============================================================================

def preprocess_digit_image(image):
    """Preprocess image for digit recognition model"""
    try:
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to 28x28 (MNIST standard)
        image = image.resize((28, 28))
        
        # Convert to numpy array
        image_array = np.array(image).astype("float32")
        
        # Normalize to 0-1 range
        image_array = image_array / 255.0
        
        # MNIST expects white digits on black background
        if np.mean(image_array) > 0.5:
            image_array = 1.0 - image_array
        
        # Ensure the digit is prominent
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array) + 1e-8)
        
        # Reshape for model input
        image_array = image_array.reshape(1, 28, 28, 1)
        
        return image_array
    except Exception as e:
        st.error(f"Image preprocessing error: {e}")
        return None

def display_digit_prediction_results(predicted_digit, confidence, probabilities, true_digit=None):
    """Display comprehensive prediction results for digits"""
    
    is_correct = true_digit is not None and predicted_digit == true_digit
    
    # Main prediction result
    if is_correct:
        st.markdown(f'<div class="success-box"><h3>🎯 Correct Prediction: {predicted_digit}</h3></div>', unsafe_allow_html=True)
    else:
        if true_digit is not None:
            st.markdown(f'<div class="warning-box"><h3>⚠️ Prediction: {predicted_digit} (Expected: {true_digit})</h3></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="success-box"><h3>🎯 Prediction: {predicted_digit}</h3></div>', unsafe_allow_html=True)
    
    # Confidence metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Confidence Score", f"{confidence:.2%}")
    with col2:
        status = "🎉 Very High" if confidence > 0.9 else "👍 High" if confidence > 0.7 else "🤔 Medium" if confidence > 0.5 else "⚠️ Low"
        st.metric("Confidence Level", status)
    with col3:
        certainty = "Excellent" if confidence > 0.8 else "Good" if confidence > 0.6 else "Fair"
        st.metric("Model Certainty", certainty)
    
    # Probability distribution
    st.subheader("📈 Probability Distribution Across Digits")
    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(range(10), probabilities, color='lightblue', alpha=0.7, edgecolor='navy', linewidth=1)
    bars[predicted_digit].set_color('gold')
    bars[predicted_digit].set_alpha(1.0)
    
    if true_digit is not None and true_digit != predicted_digit:
        bars[true_digit].set_color('red')
        bars[true_digit].set_alpha(0.8)
    
    ax.set_xlabel('Digit (0-9)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax.set_title('Model Confidence Scores for Each Digit', fontsize=14, pad=20, fontweight='bold')
    ax.set_xticks(range(10))
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)
    
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{prob:.3f}', ha='center', va='bottom', fontsize=10, 
                fontweight='bold' if i == predicted_digit else 'normal')
    
    st.pyplot(fig)

# =============================================================================
# IRIS CLASSIFICATION FUNCTIONS
# =============================================================================

def predict_iris_species(model, iris, features):
    """Predict iris species based on input features"""
    try:
        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]
        species_name = iris.target_names[prediction]
        return species_name, probabilities[prediction], probabilities
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, 0.0, []

def display_iris_analysis():
    """Display comprehensive Iris dataset analysis"""
    try:
        # Load data
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = [iris.target_names[i] for i in iris.target]
        
        st.subheader("📊 Iris Dataset Overview")
        
        # Dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Features", len(iris.feature_names))
        with col3:
            st.metric("Species", len(iris.target_names))
        
        # Data preview
        with st.expander("View Dataset Sample"):
            st.dataframe(df.head(10))
        
        # Visualization
        st.subheader("📈 Data Visualization")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Feature distributions
        features = iris.feature_names[:4]
        for i, feature in enumerate(features):
            ax = axes[i//2, i%2]
            for species in iris.target_names:
                species_data = df[df['species'] == species][feature]
                ax.hist(species_data, alpha=0.7, label=species, bins=15)
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.set_title(f'Distribution of {feature}')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Correlation heatmap
        st.subheader("🔗 Feature Correlation Heatmap")
        numeric_df = df.drop('species', axis=1)
        corr_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error in Iris analysis: {e}")

# =============================================================================
# TEXT ANALYSIS FUNCTIONS
# =============================================================================

def analyze_sentiment(text, nlp=None):
    """Analyze sentiment of text using rule-based approach"""
    positive_words = {
        "love", "amazing", "perfectly", "fantastic", "incredible", "best", 
        "highly", "great", "excellent", "awesome", "recommended", "good",
        "fantastic", "wonderful", "outstanding", "superb", "brilliant",
        "exceptional", "outstanding", "perfect", "responsive", "comfortable"
    }
    
    negative_words = {
        "terrible", "poor", "awful", "cheap", "bad", "horrible", "disappointed",
        "unhelpful", "cracked", "worst", "disappointing", "frustrating", 
        "useless", "broken", "expensive", "jams", "shorter", "stopped"
    }
    
    text_low = text.lower()
    words = text_low.split()
    pos = sum(1 for word in words if word in positive_words)
    neg = sum(1 for word in words if word in negative_words)
    
    total_words = pos + neg
    if total_words > 0:
        sentiment_score = (pos - neg) / total_words
    else:
        sentiment_score = 0
    
    if pos > neg:
        return "Positive", pos, neg, sentiment_score
    elif neg > pos:
        return "Negative", pos, neg, sentiment_score
    else:
        return "Neutral", pos, neg, sentiment_score

def extract_entities(text, nlp=None):
    """Extract named entities from text"""
    if nlp is None:
        # Basic entity extraction without spaCy
        entities = []
        text_lower = text.lower()
        
        # Simple pattern matching for common entities
        brands_products = {
            'apple': ['iphone', 'ipad', 'macbook', 'apple'],
            'samsung': ['samsung galaxy', 'samsung'],
            'google': ['google pixel', 'google'],
            'microsoft': ['microsoft surface', 'microsoft'],
            'dell': ['dell xps', 'dell'],
            'sony': ['sony headphones', 'sony'],
            'hp': ['hp printer', 'hp'],
            'lenovo': ['lenovo thinkpad', 'lenovo']
        }
        
        for brand, products in brands_products.items():
            for product in products:
                if product in text_lower:
                    entities.append((product.title(), 'PRODUCT'))
                    entities.append((brand.title(), 'BRAND'))
        
        return list(set(entities))  # Remove duplicates
    
    else:
        # Use spaCy for advanced NER
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

def display_text_analysis_results(reviews, nlp=None):
    """Display comprehensive text analysis results"""
    st.subheader("📊 Analysis Results")
    
    sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    all_entities = []
    analysis_results = []
    
    for i, review in enumerate(reviews, 1):
        # Sentiment analysis
        sentiment, pos_count, neg_count, sentiment_score = analyze_sentiment(review)
        sentiment_counts[sentiment] += 1
        
        # Entity extraction
        entities = extract_entities(review, nlp)
        all_entities.extend(entities)
        
        analysis_results.append({
            'review_id': i,
            'text': review,
            'sentiment': sentiment,
            'positive_words': pos_count,
            'negative_words': neg_count,
            'sentiment_score': sentiment_score,
            'entities': entities
        })
    
    # Display summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Reviews", len(reviews))
    with col2:
        st.metric("Positive", sentiment_counts['Positive'])
    with col3:
        st.metric("Negative", sentiment_counts['Negative'])
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Sentiment distribution
    axes[0].bar(sentiment_counts.keys(), sentiment_counts.values(), 
                color=['#2ecc71', '#e74c3c', '#95a5a6'])
    axes[0].set_title('Sentiment Distribution')
    axes[0].set_ylabel('Number of Reviews')
    
    # Entity type distribution (if spaCy is available)
    if nlp and all_entities:
        entity_types = {}
        for entity, etype in all_entities:
            entity_types[etype] = entity_types.get(etype, 0) + 1
        
        if entity_types:
            axes[1].bar(entity_types.keys(), entity_types.values(), color='#3498db')
            axes[1].set_title('Entity Type Distribution')
            axes[1].set_ylabel('Count')
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Detailed results
    st.subheader("📝 Detailed Analysis")
    for result in analysis_results:
        with st.expander(f"Review {result['review_id']}: {result['sentiment']} Sentiment"):
            st.write(f"**Text:** {result['text']}")
            st.write(f"**Sentiment:** {result['sentiment']} (Score: {result['sentiment_score']:.2f})")
            st.write(f"**Positive words:** {result['positive_words']}, **Negative words:** {result['negative_words']}")
            if result['entities']:
                st.write("**Extracted Entities:**")
                for entity, etype in result['entities']:
                    st.write(f"  - {entity} ({etype})")
    
    return analysis_results

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">🔢 Digital Vision AI</div>', unsafe_allow_html=True)
    st.markdown("### Complete Machine Learning Pipeline: Digit Recognition, Iris Classification & Text Analysis")
    
    # Initialize models
    digit_model = load_digit_model()
    iris_model, iris_data = load_iris_model()
    nlp_model = initialize_nlp()
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Navigation")
        st.markdown("Choose a module to explore:")
        
        # Module selection
        selected_tab = st.radio(
            "Select Module:",
            ["Digit Recognition", "Iris Classification", "Text Analysis"],
            index=0
        )
        
        st.header("🔧 System Status")
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.metric("Digit Model", "✅" if digit_model else "❌")
            st.metric("Iris Model", "✅" if iris_model else "❌")
        with status_col2:
            st.metric("NLP Engine", "✅" if nlp_model else "🔶")
        
        st.header("💡 Tips")
        if selected_tab == "Digit Recognition":
            st.markdown("""
            - Draw clearly in the center
            - Use thick strokes
            - White on black works best
            """)
        elif selected_tab == "Iris Classification":
            st.markdown("""
            - Enter measurements in cm
            - All features are required
            - Model accuracy: ~95%
            """)
        elif selected_tab == "Text Analysis":
            st.markdown("""
            - Enter product reviews
            - Sentiment analysis included
            - Entity extraction for brands/products
            """)
    
    # Main content based on selected tab
    if selected_tab == "Digit Recognition":
        st.markdown('<div class="sub-header">✏️ Handwritten Digit Recognition</div>', unsafe_allow_html=True)
        
        if digit_model is None:
            st.error("""
            ❌ Digit recognition model not available. Please:
            1. Run `main.py` first to train the model
            2. Ensure TensorFlow is installed
            3. Check that `models/digit_recognition_model.h5` exists
            """)
            return
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎨 Input Methods")
            
            # Drawing canvas
            try:
                from streamlit_drawable_canvas import st_canvas
                
                st.markdown("**Draw a digit (0-9):**")
                canvas_result = st_canvas(
                    fill_color="rgba(0, 0, 0, 1)",
                    stroke_width=20,
                    stroke_color="rgba(255, 255, 255, 1)",
                    background_color="rgba(0, 0, 0, 1)",
                    height=300,
                    width=300,
                    drawing_mode="freedraw",
                    key="canvas",
                    display_toolbar=True,
                )
                
                col1a, col1b = st.columns(2)
                with col1a:
                    if st.button("🗑️ Clear Canvas", use_container_width=True):
                        st.rerun()
                with col1b:
                    classify_drawing = st.button("🔍 Classify Drawing", use_container_width=True, type="primary")
                
                if classify_drawing and canvas_result and canvas_result.image_data is not None:
                    with st.spinner("Analyzing your drawing..."):
                        canvas_image = Image.fromarray((canvas_result.image_data * 255).astype('uint8'))
                        processed_image = preprocess_digit_image(canvas_image)
                        
                        if processed_image is not None:
                            prediction = digit_model.predict(processed_image, verbose=0)
                            predicted_digit = np.argmax(prediction)
                            confidence = np.max(prediction)
                            
                            with col2:
                                st.subheader("📊 Prediction Results")
                                st.image(canvas_image, caption='Your Drawing', use_container_width=True)
                                display_digit_prediction_results(predicted_digit, confidence, prediction[0])
            
            except ImportError:
                st.error("Drawing components not available. Install: `pip install streamlit-drawable-canvas`")
                
            # File upload alternative
            st.markdown("---")
            st.subheader("📤 Upload Image")
            uploaded_file = st.file_uploader("Or upload a digit image", type=["png", "jpg", "jpeg"])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                if st.button("🔍 Analyze Uploaded Image", use_container_width=True):
                    with st.spinner("Analyzing uploaded image..."):
                        processed_image = preprocess_digit_image(image)
                        if processed_image is not None:
                            prediction = digit_model.predict(processed_image, verbose=0)
                            predicted_digit = np.argmax(prediction)
                            confidence = np.max(prediction)
                            
                            with col2:
                                st.subheader("📊 Prediction Results")
                                display_digit_prediction_results(predicted_digit, confidence, prediction[0])
        
        with col2:
            if not (classify_drawing and canvas_result and canvas_result.image_data is not None) and not uploaded_file:
                st.info("👆 Draw a digit or upload an image to see predictions here")
                st.markdown("""
                <div class="info-box">
                <strong>About Digit Recognition:</strong><br><br>
                • **Model**: Convolutional Neural Network (CNN)<br>
                • **Training Data**: MNIST dataset (60,000 images)<br>
                • **Accuracy**: >98% on test data<br>
                • **Input**: 28×28 grayscale images<br>
                • **Output**: Digit classification 0-9<br>
                </div>
                """, unsafe_allow_html=True)
    
    elif selected_tab == "Iris Classification":
        st.markdown('<div class="sub-header">🌺 Iris Species Classification</div>', unsafe_allow_html=True)
        
        if iris_model is None or iris_data is None:
            st.error("Iris classification model not available.")
            return
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🔮 Species Prediction")
            st.markdown("Enter iris flower measurements to predict the species:")
            
            # Input fields for features
            sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
            sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
            petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
            petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
            
            if st.button("🌺 Predict Species", type="primary", use_container_width=True):
                features = [sepal_length, sepal_width, petal_length, petal_width]
                species, confidence, probabilities = predict_iris_species(iris_model, iris_data, features)
                
                if species:
                    st.success(f"**Predicted Species: {species}**")
                    st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Display probabilities for all species
                    st.subheader("📊 Prediction Probabilities")
                    prob_df = pd.DataFrame({
                        'Species': iris_data.target_names,
                        'Probability': probabilities
                    })
                    st.dataframe(prob_df.style.format({'Probability': '{:.2%}'}))
                    
                    # Feature importance visualization
                    st.subheader("🔍 Feature Importance")
                    feature_importance = iris_model.feature_importances_
                    fig, ax = plt.subplots(figsize=(10, 6))
                    y_pos = np.arange(len(iris_data.feature_names))
                    ax.barh(y_pos, feature_importance, color='steelblue', alpha=0.8)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(iris_data.feature_names)
                    ax.set_xlabel('Importance Score')
                    ax.set_title('Feature Importance in Classification')
                    ax.grid(axis='x', alpha=0.3)
                    st.pyplot(fig)
        
        with col2:
            st.subheader("📚 Dataset Analysis")
            display_iris_analysis()
    
    elif selected_tab == "Text Analysis":
        st.markdown('<div class="sub-header">📝 Text Analysis & Sentiment Detection</div>', unsafe_allow_html=True)
        
        st.subheader("🔍 Analyze Product Reviews")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Use Sample Reviews", "Enter Custom Text"],
            horizontal=True
        )
        
        reviews = []
        
        if input_method == "Use Sample Reviews":
            st.info("Using pre-loaded sample product reviews for analysis")
            sample_reviews = [
                "I absolutely love my new iPhone 14 Pro from Apple. The camera quality is amazing and battery life lasts all day!",
                "This Samsung Galaxy S23 is terrible. The screen cracked after one week and customer service was unhelpful.",
                "My Dell XPS laptop from Amazon works perfectly for programming and gaming. Highly recommended!",
                "The Sony headphones I bought have poor sound quality and the build feels cheap. Very disappointed.",
                "Google Pixel 7 has an incredible camera and clean Android experience. Best phone I've ever owned!"
            ]
            reviews = sample_reviews
            
            for i, review in enumerate(reviews, 1):
                st.write(f"**Review {i}:** {review}")
        
        else:
            st.text_area("Enter your text for analysis:", key="custom_text", height=150)
            if st.session_state.custom_text:
                reviews = [st.session_state.custom_text]
            else:
                st.warning("Please enter some text to analyze.")
        
        if reviews and st.button("🔍 Analyze Text", type="primary", use_container_width=True):
            with st.spinner("Analyzing text..."):
                analysis_results = display_text_analysis_results(reviews, nlp_model)
        
        # Additional NLP information
        with st.expander("ℹ️ About Text Analysis"):
            st.markdown("""
            **Features Included:**
            
            **1. Sentiment Analysis**
            - Rule-based approach using positive/negative word dictionaries
            - Calculates sentiment score (-1 to +1)
            - Classifies as Positive, Negative, or Neutral
            
            **2. Named Entity Recognition (NER)**
            - Extracts product names and brands
            - Uses spaCy for advanced entity recognition
            - Fallback to pattern matching if spaCy not available
            
            **3. Entity Types:**
            - PRODUCT: Product names and models
            - BRAND: Company and brand names
            - ORG: Organizations
            - PERSON: People names
            - And more...
            
            **Technical Details:**
            - Built with spaCy for advanced NLP
            - Custom sentiment analysis rules
            - Expandable entity recognition patterns
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <h3>🔍 Digital Vision AI - Complete ML Pipeline</h3>
    <p><strong>Modules:</strong> Digit Recognition • Iris Classification • Text Analysis</p>
    <p><small>Built with: TensorFlow • Scikit-learn • spaCy • Streamlit</small></p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()