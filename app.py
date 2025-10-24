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
    .mini-tab {
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    .prediction-result {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .reset-button {
        background-color: #dc3545 !important;
        color: white !important;
        border: 1px solid #dc3545 !important;
    }
    .reset-button:hover {
        background-color: #c82333 !important;
        border-color: #bd2130 !important;
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

@st.cache_resource
def initialize_nlp():
    """Initialize NLP components with better error handling"""
    try:
        import spacy
        from spacy.matcher import PhraseMatcher
        
        # Try to load spaCy model with multiple fallbacks
        try:
            nlp = spacy.load("en_core_web_sm")
            st.sidebar.success("✅ spaCy model loaded")
            return nlp
        except OSError:
            try:
                # Try alternative model
                nlp = spacy.load("en_core_web_lg")
                st.sidebar.success("✅ spaCy large model loaded")
                return nlp
            except OSError:
                st.sidebar.info("ℹ️ spaCy model not found. Using enhanced basic text analysis.")
                return None
                
    except ImportError:
        st.sidebar.info("ℹ️ spaCy not available. Using enhanced basic text analysis.")
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

def create_sample_digit(digit):
    """Create a clean sample digit image for demonstration"""
    try:
        # Create a black background
        img = Image.new('L', (28, 28), color=0)
        draw = ImageDraw.Draw(img)
        
        # Define digit drawing patterns
        digit_patterns = {
            0: [(7, 5), (21, 5), (24, 10), (24, 18), (21, 23), (7, 23), (4, 18), (4, 10), (7, 5)],
            1: [(14, 5), (14, 23)],
            2: [(5, 5), (20, 5), (23, 10), (20, 15), (5, 15), (5, 23), (23, 23)],
            3: [(5, 5), (20, 5), (23, 10), (20, 14), (23, 18), (20, 23), (5, 23)],
            4: [(20, 5), (20, 23), (5, 15), (23, 15)],
            5: [(23, 5), (5, 5), (5, 14), (23, 14), (23, 23), (5, 23)],
            6: [(20, 5), (5, 5), (5, 23), (23, 23), (23, 14), (5, 14)],
            7: [(5, 5), (23, 5), (15, 23)],
            8: [(5, 5), (23, 5), (23, 23), (5, 23), (5, 5), (5, 14), (23, 14)],
            9: [(23, 5), (23, 23), (5, 23), (5, 14), (23, 14)]
        }
        
        if digit in digit_patterns:
            points = digit_patterns[digit]
            if len(points) > 1:
                draw.line(points, fill=255, width=2)
        
        return img
    except Exception as e:
        st.error(f"Error creating sample digit: {e}")
        return None

def get_image_download_link(img, filename="digit.png", text="📥 Download Image"):
    """Generate a download link for the image"""
    try:
        if img is None:
            return "❌ No image available to download"
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        href = f'<a href="data:file/png;base64,{img_str}" download="{filename}" style="display: inline-block; padding: 0.5rem 1rem; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 0.3rem; margin: 0.2rem;">{text}</a>'
        return href
    except Exception as e:
        return f"❌ Error creating download link: {e}"

def display_digit_prediction_results(predicted_digit, confidence, probabilities, true_digit=None):
    """Display comprehensive prediction results for digits"""
    
    is_correct = true_digit is not None and predicted_digit == true_digit
    
    # Main prediction result
    st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
    
    if is_correct:
        st.markdown(f'<h3 style="color: #28a745;">🎯 Correct Prediction: {predicted_digit}</h3>', unsafe_allow_html=True)
    else:
        if true_digit is not None:
            st.markdown(f'<h3 style="color: #dc3545;">⚠️ Prediction: {predicted_digit} (Expected: {true_digit})</h3>', unsafe_allow_html=True)
        else:
            st.markdown(f'<h3 style="color: #28a745;">🎯 Prediction: {predicted_digit}</h3>', unsafe_allow_html=True)
    
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
    
    st.markdown('</div>', unsafe_allow_html=True)
    
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
    
    # Detailed insights
    st.subheader("💡 Analysis Insights")
    
    if is_correct:
        st.success("""
        **🎉 Excellent Recognition!**  
        • The model correctly identified the digit  
        • High confidence indicates clear input  
        • The preprocessing worked effectively
        """)
    elif true_digit is not None:
        st.error("""
        **❌ Incorrect Recognition**  
        • The model misclassified the digit  
        • This could be due to:  
          - Unclear drawing style  
          - Preprocessing differences  
          - Model confusion with similar digits
        """)
    elif confidence > 0.9:
        st.success("""
        **🎉 Excellent Recognition!**  
        • The model is highly confident in this prediction  
        • The digit appears clear and well-defined  
        • Minimal ambiguity in classification
        """)
    elif confidence > 0.7:
        st.warning("""
        **👍 Good Recognition**  
        • The model is confident in the prediction  
        • Some minor ambiguity may exist  
        • Consider redrawing for maximum accuracy
        """)
    elif confidence > 0.5:
        st.info("""
        **🤔 Moderate Confidence**  
        • The model shows some uncertainty  
        • Multiple digits had similar probabilities  
        • Try drawing more clearly or using a different style
        """)
    else:
        st.error("""
        **⚠️ Low Confidence**  
        • The model is uncertain about this digit  
        • Significant ambiguity in classification  
        • The drawing may be unclear or ambiguous
        """)
    
    # Top 3 predictions
    top_indices = np.argsort(probabilities)[-3:][::-1]
    st.subheader("🏆 Top 3 Predictions")
    
    for i, idx in enumerate(top_indices):
        emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        prob = probabilities[idx]
        is_true = (true_digit is not None and idx == true_digit)
        highlight = "**" if is_true else ""
        st.write(f"{emoji} {highlight}Digit {idx}: {prob:.3f} ({prob:.1%} confidence){highlight}")

def clear_canvas():
    """Clear the canvas by resetting the session state"""
    # Clear all canvas-related session state variables
    canvas_keys = [key for key in st.session_state.keys() if 'canvas' in key.lower()]
    for key in canvas_keys:
        del st.session_state[key]
    
    # Also clear drawing results
    if 'drawing_result' in st.session_state:
        del st.session_state.drawing_result
    
    st.success("✅ Canvas cleared!")
    st.rerun()

def reset_all():
    """Reset all session state variables including canvas"""
    keys_to_keep = ['model', 'iris_model', 'iris_data', 'nlp_model']  # Keep model references
    
    # Clear all canvas-related session state variables
    canvas_keys = [key for key in st.session_state.keys() if 'canvas' in key.lower()]
    for key in canvas_keys:
        if key not in keys_to_keep:
            del st.session_state[key]
    
    # Clear all other user data
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep and 'canvas' not in key.lower():
            del st.session_state[key]
    
    st.success("🔄 All inputs, drawings, and results have been reset!")
    st.rerun()

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
        # Enhanced basic entity extraction without spaCy
        entities = []
        text_lower = text.lower()
        
        # Enhanced pattern matching for common entities
        brands_products = {
            'APPLE': ['iphone', 'ipad', 'macbook', 'apple', 'mac', 'ios'],
            'SAMSUNG': ['samsung galaxy', 'samsung', 'galaxy', 'android'],
            'GOOGLE': ['google pixel', 'google', 'pixel', 'android'],
            'MICROSOFT': ['microsoft surface', 'microsoft', 'surface', 'windows'],
            'DELL': ['dell xps', 'dell', 'xps', 'laptop'],
            'SONY': ['sony headphones', 'sony', 'playstation'],
            'HP': ['hp printer', 'hp', 'printer', 'laptop'],
            'LENOVO': ['lenovo thinkpad', 'lenovo', 'thinkpad']
        }
        
        for brand, products in brands_products.items():
            for product in products:
                if product in text_lower:
                    entities.append((product.title(), 'PRODUCT'))
                    entities.append((brand, 'BRAND'))
        
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
    
    # Entity type distribution
    if all_entities:
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
            else:
                st.write("**Extracted Entities:** No entities found")
    
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
        
        # Reset All Button at the bottom of sidebar
        st.markdown("---")
        st.header("🛠️ Actions")
        if st.button("🔄 Reset All", use_container_width=True, type="secondary", key="reset_all_main"):
            reset_all()
    
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
        
        # Mini navigation tabs for Digit Recognition
        digit_tab1, digit_tab2, digit_tab3 = st.tabs(["✏️ Drawing Tab", "📤 Upload Tab", "🎲 Sample Tab"])
        
        with digit_tab1:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🎨 Draw Your Digit")
                st.markdown("**Draw a digit (0-9) in the canvas below:**")
                
                try:
                    from streamlit_drawable_canvas import st_canvas
                    
                    # Canvas configuration
                    canvas_result = st_canvas(
                        fill_color="rgba(0, 0, 0, 1)",
                        stroke_width=20,
                        stroke_color="rgba(255, 255, 255, 1)",
                        background_color="rgba(0, 0, 0, 1)",
                        height=300,
                        width=300,
                        drawing_mode="freedraw",
                        key="canvas_draw",
                        display_toolbar=True,
                    )
                    
                    # Canvas controls
                    col1a, col1b = st.columns(2)
                    with col1a:
                        if st.button("🗑️ Clear Canvas", use_container_width=True, key="clear_draw"):
                            clear_canvas()
                    with col1b:
                        classify_drawing = st.button("🔍 Classify Drawing", use_container_width=True, type="primary", key="classify_draw")
                    
                    # Process drawing
                    if classify_drawing and canvas_result and canvas_result.image_data is not None:
                        with st.spinner("🔄 Analyzing your drawing..."):
                            canvas_image = Image.fromarray((canvas_result.image_data * 255).astype('uint8'))
                            processed_image = preprocess_digit_image(canvas_image)
                            
                            if processed_image is not None:
                                prediction = digit_model.predict(processed_image, verbose=0)
                                predicted_digit = np.argmax(prediction)
                                confidence = np.max(prediction)
                                
                                # Store results for display in the right column
                                st.session_state.drawing_result = {
                                    'predicted_digit': predicted_digit,
                                    'confidence': confidence,
                                    'probabilities': prediction[0],
                                    'image': canvas_image
                                }
                                st.success("✅ Analysis complete! View results in the right panel.")
                    
                    # Drawing tips
                    with st.expander("💡 Drawing Tips"):
                        st.markdown("""
                        **For Best Results:**
                        - Draw clearly in the center of the canvas
                        - Use thick, white strokes on black background
                        - Make the digit fill most of the canvas
                        - Avoid faint or blurry lines
                        - Common confusions: 3 vs 8, 5 vs 6, 7 vs 1
                        """)
                
                except ImportError:
                    st.error("Drawing components not available. Install: `pip install streamlit-drawable-canvas`")
            
            with col2:
                st.subheader("📊 Prediction Results")
                
                # Display drawing results
                if 'drawing_result' in st.session_state and st.session_state.drawing_result is not None:
                    result = st.session_state.drawing_result
                    if result['image'] is not None:
                        st.image(result['image'], caption='🎨 Your Drawing', use_container_width=True)
                        display_digit_prediction_results(
                            result['predicted_digit'], 
                            result['confidence'], 
                            result['probabilities']
                        )
                else:
                    st.info("👆 Draw a digit and click 'Classify Drawing' to see predictions here")
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
        
        with digit_tab2:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📤 Upload Digit Image")
                st.markdown("**Upload a digit image for classification:**")
                
                uploaded_file = st.file_uploader(
                    "Choose a digit image (PNG, JPG, JPEG)", 
                    type=["png", "jpg", "jpeg"],
                    key="upload_digit"
                )
                
                if uploaded_file is not None:
                    try:
                        image = Image.open(uploaded_file)
                        st.image(image, caption="📷 Uploaded Image", use_container_width=True)
                        
                        if st.button("🔍 Analyze Uploaded Image", use_container_width=True, type="primary"):
                            with st.spinner("🔄 Analyzing uploaded image..."):
                                processed_image = preprocess_digit_image(image)
                                if processed_image is not None:
                                    prediction = digit_model.predict(processed_image, verbose=0)
                                    predicted_digit = np.argmax(prediction)
                                    confidence = np.max(prediction)
                                    
                                    # Store results for display
                                    st.session_state.upload_result = {
                                        'predicted_digit': predicted_digit,
                                        'confidence': confidence,
                                        'probabilities': prediction[0],
                                        'image': image
                                    }
                                    st.success("✅ Analysis complete! View results in the right panel.")
                                
                    except Exception as e:
                        st.error(f"❌ Error loading image: {e}")
                
                # Upload tips
                with st.expander("💡 Upload Tips"):
                    st.markdown("""
                    **Image Requirements:**
                    - Supported formats: PNG, JPG, JPEG
                    - White digits on dark background work best
                    - Clear, centered digits give best results
                    - The image will be automatically resized to 28x28 pixels
                    """)
            
            with col2:
                st.subheader("📊 Prediction Results")
                
                # Display upload results
                if 'upload_result' in st.session_state and st.session_state.upload_result is not None:
                    result = st.session_state.upload_result
                    if result['image'] is not None:
                        st.image(result['image'], caption='📷 Uploaded Image', use_container_width=True)
                        display_digit_prediction_results(
                            result['predicted_digit'], 
                            result['confidence'], 
                            result['probabilities']
                        )
                else:
                    st.info("👆 Upload an image and click 'Analyze Uploaded Image' to see predictions here")
                    st.markdown("""
                    <div class="info-box">
                    <strong>Upload Features:</strong><br><br>
                    • **Supported Formats**: PNG, JPG, JPEG<br>
                    • **Automatic Preprocessing**: Resize & normalization<br>
                    • **Real-time Analysis**: Instant classification<br>
                    • **Confidence Scoring**: See how sure the model is<br>
                    </div>
                    """, unsafe_allow_html=True)
        
        with digit_tab3:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🎲 Sample Digit Testing")
                st.markdown("**Generate and test sample digits:**")
                
                # Sample digit selection
                col1a, col1b = st.columns([2, 1])
                with col1a:
                    sample_digit = st.selectbox(
                        "Select a digit to generate:",
                        options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        index=3,
                        help="Choose a digit to generate a sample image"
                    )
                with col1b:
                    st.write("")  # Spacer
                    st.write("")  # Spacer
                    generate_sample = st.button("🎯 Generate Sample", use_container_width=True)
                
                # Generate and display sample
                if generate_sample:
                    sample_image = create_sample_digit(sample_digit)
                    if sample_image is not None:
                        st.session_state.sample_image = sample_image
                        st.session_state.sample_digit = sample_digit
                        st.success(f"✅ Generated sample digit: {sample_digit}")
                
                # Display and analyze sample
                if 'sample_image' in st.session_state:
                    st.image(st.session_state.sample_image, 
                            caption=f'🎲 Generated Sample: Digit {st.session_state.sample_digit}', 
                            width=200)
                    
                    # Analysis buttons
                    col3, col4 = st.columns(2)
                    with col3:
                        if st.button("🔍 Analyze Sample", use_container_width=True, type="primary"):
                            if digit_model:
                                with st.spinner("🔄 Analyzing sample digit..."):
                                    processed_image = preprocess_digit_image(st.session_state.sample_image)
                                    if processed_image is not None:
                                        prediction = digit_model.predict(processed_image, verbose=0)
                                        predicted_digit = np.argmax(prediction)
                                        confidence = np.max(prediction)
                                        
                                        # Store results for display
                                        st.session_state.sample_result = {
                                            'predicted_digit': predicted_digit,
                                            'confidence': confidence,
                                            'probabilities': prediction[0],
                                            'image': st.session_state.sample_image,
                                            'true_digit': st.session_state.sample_digit
                                        }
                                        st.success("✅ Analysis complete! View results in the right panel.")
                    with col4:
                        if st.session_state.sample_image is not None:
                            st.markdown(get_image_download_link(
                                st.session_state.sample_image,
                                f"sample_digit_{st.session_state.sample_digit}.png",
                                "📥 Download Sample"
                            ), unsafe_allow_html=True)
                
                # Sample testing info
                with st.expander("ℹ️ About Sample Testing"):
                    st.markdown("""
                    **Sample Testing Features:**
                    - Generate clean digit samples for testing
                    - Verify model accuracy with known digits
                    - Download samples for reference
                    - Test model performance across all digits
                    """)
            
            with col2:
                st.subheader("📊 Prediction Results")
                
                # Display sample results
                if 'sample_result' in st.session_state and st.session_state.sample_result is not None:
                    result = st.session_state.sample_result
                    if result['image'] is not None:
                        st.image(result['image'], 
                                caption=f'🎲 Sample Digit: {result["true_digit"]}', 
                                use_container_width=True)
                        display_digit_prediction_results(
                            result['predicted_digit'], 
                            result['confidence'], 
                            result['probabilities'],
                            result['true_digit']
                        )
                else:
                    st.info("👆 Generate a sample digit and click 'Analyze Sample' to see predictions here")
                    st.markdown("""
                    <div class="info-box">
                    <strong>Sample Testing Benefits:</strong><br><br>
                    • **Accuracy Verification**: Test with known digits<br>
                    • **Model Validation**: Ensure model works correctly<br>
                    • **Performance Testing**: Check across all digit types<br>
                    • **Reference Samples**: Download for comparison<br>
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
            custom_text = st.text_area("Enter your text for analysis:", key="custom_text", height=150,
                                      placeholder="Paste your product review or any text here...")
            if custom_text:
                reviews = [custom_text]
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
            - Enhanced pattern matching if spaCy not available
            
            **3. Entity Types:**
            - PRODUCT: Product names and models
            - BRAND: Company and brand names
            - ORG: Organizations
            - PERSON: People names
            
            **Technical Details:**
            - Enhanced basic analysis when spaCy not available
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