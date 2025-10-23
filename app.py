"""
Digital Vision AI - Interactive Web Application
Real-time digit recognition with drawing interface
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import matplotlib.pyplot as plt
import os
import io
import base64

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
    .canvas-container {
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 10px;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
        margin: 2px 0;
    }
    .download-link {
        display: block;
        text-align: center;
        padding: 10px;
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        text-decoration: none;
        margin: 5px 0;
    }
    .download-link:hover {
        background-color: #0056b3;
        color: white;
        text-decoration: none;
    }
    .input-section {
        border-right: 2px solid #e0e0e0;
        padding-right: 20px;
    }
    .results-section {
        padding-left: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_digit_model():
    """Load the pre-trained digit recognition model with better error handling"""
    try:
        # Import TensorFlow only when needed
        import tensorflow as tf
        
        # Define model path - use relative path for deployment
        model_path = 'models/digit_recognition_model.h5'
        
        # Try to load the model directly without checking file existence first
        # This works better in cloud environments
        try:
            model = tf.keras.models.load_model(model_path)
            st.sidebar.success("✅ Model loaded successfully!")
            return model
        except (OSError, IOError) as e:
            st.sidebar.error(f"🔍 Model file not found at: {model_path}")
            st.sidebar.info("""
            **To fix this:**
            1. Run `python main.py` locally to train the model
            2. Upload the model file to GitHub
            3. Redeploy on Streamlit Cloud
            """)
            
            # Provide more detailed error information
            st.sidebar.error(f"Detailed error: {str(e)}")
            return None
            
    except ImportError:
        st.sidebar.error("❌ TensorFlow not available")
        st.sidebar.info("Install with: pip install tensorflow")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Unexpected error loading model: {e}")
        return None

def preprocess_digit_image(image):
    """Preprocess image for digit recognition model - FIXED for correct prediction"""
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
        # If the image has light background (mean > 0.5), invert it
        if np.mean(image_array) > 0.5:
            image_array = 1.0 - image_array
        
        # Ensure the digit is prominent - enhance contrast
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array) + 1e-8)
        
        # Reshape for model input (batch_size, height, width, channels)
        image_array = image_array.reshape(1, 28, 28, 1)
        
        return image_array
    except Exception as e:
        st.error(f"❌ Image preprocessing error: {e}")
        return None

def enhance_digit_image(image):
    """Enhance digit image for better recognition using PIL only"""
    try:
        # Convert to numpy array for processing
        img_array = np.array(image)
        
        # Simple contrast enhancement using PIL
        if image.mode == 'L':
            # Use PIL's built-in enhancement
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            enhanced_image = enhancer.enhance(2.0)  # Increase contrast
        else:
            enhanced_image = image
        
        return enhanced_image
    except Exception as e:
        st.warning(f"⚠️ Image enhancement failed: {e}")
        return image

def create_handwritten_digit(digit):
    """Create a proper handwritten-like digit for testing"""
    try:
        # Create a black background
        img = Image.new('L', (28, 28), color=0)
        draw = ImageDraw.Draw(img)
        
        # Define digit drawing patterns (simplified for better recognition)
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
            # Draw the digit with white color (255)
            if len(points) > 1:
                draw.line(points, fill=255, width=2)
        
        return img
    except Exception as e:
        st.error(f"❌ Error creating handwritten digit: {e}")
        return create_sample_digit(digit)

def create_sample_digit(digit):
    """Generate a clean sample digit image for demonstration"""
    try:
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.text(0.5, 0.5, str(digit), fontsize=120, ha='center', va='center', 
                color='white', weight='bold')
        ax.set_facecolor('black')
        ax.axis('off')
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, 
                    facecolor='black', dpi=100)
        buf.seek(0)
        image = Image.open(buf)
        plt.close()
        
        return image
    except Exception as e:
        st.error(f"❌ Error creating sample digit: {e}")
        return None

def get_image_download_link(img, filename="digit.png", text="📥 Download Drawing"):
    """Generate a download link for the image"""
    try:
        if img is None:
            return "❌ No image available to download"
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        href = f'<a href="data:file/png;base64,{img_str}" download="{filename}" class="download-link">{text}</a>'
        return href
    except Exception as e:
        return f"❌ Error creating download link: {e}"

def clear_canvas():
    """Clear the canvas session state"""
    if 'canvas_data' in st.session_state:
        del st.session_state.canvas_data
    if 'drawing_result' in st.session_state:
        del st.session_state.drawing_result
    st.success("✅ Canvas cleared!")

def initialize_session_state():
    """Initialize session state variables"""
    if 'last_drawing' not in st.session_state:
        st.session_state.last_drawing = None
    if 'sample_image' not in st.session_state:
        st.session_state.sample_image = None
    if 'sample_digit' not in st.session_state:
        st.session_state.sample_digit = 3
    if 'sample_result' not in st.session_state:
        st.session_state.sample_result = None
    if 'upload_result' not in st.session_state:
        st.session_state.upload_result = None
    if 'drawing_result' not in st.session_state:
        st.session_state.drawing_result = None

def display_prediction_results(predicted_digit, confidence, probabilities, true_digit=None, input_type="input"):
    """Display comprehensive prediction results"""
    
    # Determine if prediction is correct
    is_correct = true_digit is not None and predicted_digit == true_digit
    
    # Main prediction result
    if is_correct:
        st.markdown(f'<div class="success-box"><h3>🎯 Correct Prediction: {predicted_digit}</h3></div>', unsafe_allow_html=True)
    else:
        if true_digit is not None:
            st.markdown(f'<div class="warning-box"><h3>⚠️ Prediction: {predicted_digit} (Expected: {true_digit})</h3></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="success-box"><h3>🎯 Prediction: {predicted_digit}</h3></div>', unsafe_allow_html=True)
    
    # Confidence metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Confidence Score", f"{confidence:.2%}")
    
    with col2:
        if confidence > 0.9:
            status = "🎉 Very High"
        elif confidence > 0.7:
            status = "👍 High" 
        elif confidence > 0.5:
            status = "🤔 Medium"
        else:
            status = "⚠️ Low"
        st.metric("Confidence Level", status)
    
    with col3:
        certainty = "Excellent" if confidence > 0.8 else "Good" if confidence > 0.6 else "Fair"
        st.metric("Model Certainty", certainty)
    
    # Probability distribution chart
    st.subheader("📈 Probability Distribution Across Digits")
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Create bars for all digits
    bars = ax.bar(range(10), probabilities, color='lightblue', alpha=0.7, edgecolor='navy', linewidth=1)
    
    # Highlight the predicted digit
    bars[predicted_digit].set_color('gold')
    bars[predicted_digit].set_alpha(1.0)
    bars[predicted_digit].set_edgecolor('darkorange')
    bars[predicted_digit].set_linewidth(2)
    
    # Highlight true digit if available
    if true_digit is not None and true_digit != predicted_digit:
        bars[true_digit].set_color('red')
        bars[true_digit].set_alpha(0.8)
        bars[true_digit].set_edgecolor('darkred')
        bars[true_digit].set_linewidth(2)
    
    ax.set_xlabel('Digit (0-9)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax.set_title('Model Confidence Scores for Each Digit', fontsize=14, pad=20, fontweight='bold')
    ax.set_xticks(range(10))
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{prob:.3f}', ha='center', va='bottom', fontsize=10, 
                fontweight='bold' if i == predicted_digit else 'normal',
                color='black' if i != predicted_digit else 'darkred')
    
    st.pyplot(fig, use_container_width=True)
    
    # Detailed insights based on confidence
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

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🔢 Digital Vision AI</div>', unsafe_allow_html=True)
    st.markdown("### Intelligent Handwritten Digit Recognition System")
    
    # Load model
    model = load_digit_model()
    
    # Debug information
    if model is None:
        st.sidebar.warning("🔧 Debug Info: Model is None")
        # Add debug information about current directory
        try:
            st.sidebar.write("📁 Current directory:", os.getcwd())
            if os.path.exists('models'):
                st.sidebar.write("📁 Models directory contents:", os.listdir('models'))
            else:
                st.sidebar.write("❌ Models directory not found")
        except Exception as e:
            st.sidebar.write(f"❌ Debug error: {e}")
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 System Information")
        
        if model:
            st.markdown("""
            <div class="success-box">
            <strong>✅ Model Status: Ready</strong><br>
            • Accuracy: 98.2%<br>
            • Training: 60,000 images<br>
            • Architecture: CNN<br>
            • Input: 28×28 grayscale<br>
            • Output: Digit 0-9
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
            <strong>⚠️ Model Status: Not Available</strong><br>
            • Please run main.py first<br>
            • Install TensorFlow if needed<br>
            • Check model file exists
            </div>
            """, unsafe_allow_html=True)
        
        st.header("🛠️ How to Use")
        st.markdown("""
        1. **Draw** a digit in the canvas
        2. **Click** 'Classify Drawing'
        3. **View** real-time predictions
        4. **Download** your drawing
        5. **Upload** images for testing
        """)
        
        st.header("💡 Drawing Tips for Better Accuracy")
        st.markdown("""
        - **Draw clearly** in the center
        - **Use thick strokes** (20px recommended)
        - **White on black** background
        - **Fill most of the canvas**
        - **Avoid blurry** or faint lines
        - **Common confusions**:
          - 3 vs 8, 5 vs 6, 7 vs 1
        """)
        
        # Quick actions
        st.header("⚡ Quick Actions")
        if st.button("🔄 Reset All", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key != 'model':  # Don't clear the cached model
                    del st.session_state[key]
            st.rerun()

    # Main content area - Split into two columns for input and results
    col1, col2 = st.columns([1, 1])
    
    # LEFT COLUMN: Input Methods
    with col1:
        st.markdown('<div class="sub-header input-section">🎨 Input Methods</div>', unsafe_allow_html=True)
        
        # Tab interface for different input methods
        tab1, tab2, tab3 = st.tabs(["✏️ Draw Digit", "📤 Upload Image", "🎲 Sample Digits"])
        
        with tab1:
            st.markdown("#### Draw Your Digit")
            st.markdown("**Tip**: Draw clearly in the center with thick strokes")
            st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
            
            try:
                from streamlit_drawable_canvas import st_canvas
                
                # Canvas configuration
                canvas_result = st_canvas(
                    fill_color="rgba(0, 0, 0, 1)",  # Black background
                    stroke_width=20,  # Thick strokes for better recognition
                    stroke_color="rgba(255, 255, 255, 1)",  # White drawing
                    background_color="rgba(0, 0, 0, 1)",  # Black background
                    height=300,
                    width=300,
                    drawing_mode="freedraw",
                    key="canvas",
                    display_toolbar=True,
                )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Canvas controls
                col1a, col1b = st.columns(2)
                with col1a:
                    if st.button("🗑️ Clear Canvas", use_container_width=True, type="secondary"):
                        clear_canvas()
                        st.rerun()
                
                with col1b:
                    classify_drawing = st.button("🔍 Classify Drawing", use_container_width=True, type="primary")
                
                # Process drawing when classify button is clicked
                if classify_drawing and canvas_result and canvas_result.image_data is not None:
                    try:
                        # Convert canvas data to image
                        canvas_image = Image.fromarray((canvas_result.image_data * 255).astype('uint8'))
                        
                        if model:
                            with st.spinner("🔄 Analyzing your drawing..."):
                                # Enhance the drawing
                                enhanced_image = enhance_digit_image(canvas_image)
                                processed_image = preprocess_digit_image(enhanced_image)
                                if processed_image is not None:
                                    prediction = model.predict(processed_image, verbose=0)
                                    predicted_digit = np.argmax(prediction)
                                    confidence = np.max(prediction)
                                    
                                    # Store results in session state
                                    st.session_state.drawing_result = {
                                        'predicted_digit': predicted_digit,
                                        'confidence': confidence,
                                        'probabilities': prediction[0],
                                        'image': canvas_image,
                                        'enhanced_image': enhanced_image
                                    }
                                    st.success("✅ Analysis complete! View results in the right panel.")
                                else:
                                    st.error("❌ Failed to process the drawing image")
                        else:
                            st.error("❌ Model not available for analysis")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing drawing: {e}")
                        st.info("💡 Please ensure you've drawn a clear digit in the center of the canvas")
                
                # Save and download functionality
                if canvas_result and canvas_result.image_data is not None:
                    col1c, col1d = st.columns(2)
                    with col1c:
                        if st.button("💾 Save Drawing", use_container_width=True):
                            drawing_img = Image.fromarray((canvas_result.image_data * 255).astype('uint8'))
                            st.session_state.last_drawing = drawing_img
                            st.success("✅ Drawing saved!")
                    
                    with col1d:
                        if st.session_state.last_drawing is not None:
                            st.markdown(get_image_download_link(
                                st.session_state.last_drawing, 
                                "my_digit.png", 
                                "📥 Download Drawing"
                            ), unsafe_allow_html=True)
                
            except ImportError:
                st.markdown('</div>', unsafe_allow_html=True)
                st.error("❌ Drawing components not available")
                st.info("💡 Install required package: `pip install streamlit-drawable-canvas`")
                canvas_result = None
                classify_drawing = False
        
        with tab2:
            st.markdown("#### Upload Digit Image")
            st.markdown("**Supported formats**: PNG, JPG, JPEG")
            uploaded_file = st.file_uploader(
                "Choose an image file containing a handwritten digit",
                type=["png", "jpg", "jpeg"],
                help="The image will be automatically processed and enhanced for better recognition."
            )
            
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="📷 Uploaded Image", use_container_width=True)
                    st.success("✅ Image uploaded successfully!")
                    
                    # Auto-classify uploaded image
                    if st.button("🔍 Analyze Uploaded Image", use_container_width=True, type="primary"):
                        if model:
                            with st.spinner("🔄 Analyzing uploaded image..."):
                                # Enhance image first
                                enhanced_image = enhance_digit_image(image)
                                processed_image = preprocess_digit_image(enhanced_image)
                                if processed_image is not None:
                                    prediction = model.predict(processed_image, verbose=0)
                                    predicted_digit = np.argmax(prediction)
                                    confidence = np.max(prediction)
                                    
                                    # Store results
                                    st.session_state.upload_result = {
                                        'predicted_digit': predicted_digit,
                                        'confidence': confidence,
                                        'probabilities': prediction[0],
                                        'image': enhanced_image
                                    }
                                    st.success("✅ Analysis complete! View results in the right panel.")
                        else:
                            st.error("❌ Model not available for analysis")
                    
                except Exception as e:
                    st.error(f"❌ Error loading image: {e}")
        
        with tab3:
            st.markdown("#### Try Sample Digits")
            st.markdown("Test the model with pre-generated digits")
            
            # Sample digit selection
            sample_digit = st.selectbox(
                "Select a digit to generate:",
                options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                index=3,
                help="Choose a digit to generate a sample image"
            )
            
            # Action buttons
            col3a, col3b = st.columns(2)
            
            with col3a:
                if st.button("🎯 Generate Sample", use_container_width=True):
                    sample_image = create_handwritten_digit(sample_digit)
                    if sample_image is not None:
                        st.session_state.sample_image = sample_image
                        st.session_state.sample_digit = sample_digit
                        st.success(f"✅ Generated sample digit: {sample_digit}")
                    else:
                        st.error("❌ Failed to generate sample image")
            
            with col3b:
                if st.button("🔍 Analyze Sample", use_container_width=True, type="primary"):
                    if st.session_state.sample_image is not None and model:
                        with st.spinner("🔄 Analyzing sample digit..."):
                            try:
                                processed_image = preprocess_digit_image(st.session_state.sample_image)
                                if processed_image is not None:
                                    prediction = model.predict(processed_image, verbose=0)
                                    predicted_digit = np.argmax(prediction)
                                    confidence = np.max(prediction)
                                    
                                    st.session_state.sample_result = {
                                        'predicted_digit': predicted_digit,
                                        'confidence': confidence,
                                        'probabilities': prediction[0],
                                        'image': st.session_state.sample_image,
                                        'true_digit': st.session_state.sample_digit
                                    }
                                    st.success("✅ Sample analyzed successfully! View results in the right panel.")
                                else:
                                    st.error("❌ Failed to process sample image")
                                
                            except Exception as e:
                                st.error(f"❌ Error analyzing sample: {e}")
                    else:
                        if st.session_state.sample_image is None:
                            st.error("❌ Please generate a sample first")
                        else:
                            st.error("❌ Model not available for analysis")
            
            # Display generated sample
            if st.session_state.sample_image is not None:
                try:
                    st.image(st.session_state.sample_image, 
                            caption=f'🎲 Generated Sample: Digit {st.session_state.sample_digit}', 
                            width=200)
                    
                    # Download sample
                    if st.button("📥 Download Sample", use_container_width=True):
                        st.markdown(get_image_download_link(
                            st.session_state.sample_image,
                            f"sample_digit_{st.session_state.sample_digit}.png",
                            "📥 Download Sample Image"
                        ), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Error displaying sample image: {e}")
                    # Reset the problematic image
                    st.session_state.sample_image = None

    # RIGHT COLUMN: Prediction Results (directly opposite input methods)
    with col2:
        st.markdown('<div class="sub-header results-section">📊 Prediction Results</div>', unsafe_allow_html=True)
        
        # Display drawing results
        if st.session_state.drawing_result is not None and model:
            result = st.session_state.drawing_result
            if result['image'] is not None:
                st.image(result['image'], caption='🎨 Your Drawing', use_container_width=True)
                display_prediction_results(
                    result['predicted_digit'], 
                    result['confidence'], 
                    result['probabilities'], 
                    None,
                    "drawing"
                )
        
        # Display uploaded file results
        elif st.session_state.upload_result is not None and model:
            result = st.session_state.upload_result
            if result['image'] is not None:
                st.image(result['image'], caption='📷 Uploaded Image', use_container_width=True)
                display_prediction_results(
                    result['predicted_digit'], 
                    result['confidence'], 
                    result['probabilities'], 
                    None,
                    "uploaded image"
                )
        
        # Display sample results
        elif st.session_state.sample_result is not None and model:
            result = st.session_state.sample_result
            if result['image'] is not None:
                try:
                    st.image(result['image'], 
                            caption=f'🎲 Sample Digit: {result["true_digit"]}', 
                            use_container_width=True)
                    display_prediction_results(
                        result['predicted_digit'], 
                        result['confidence'], 
                        result['probabilities'], 
                        result['true_digit'],
                        "sample digit"
                    )
                except Exception as e:
                    st.error(f"❌ Error displaying sample result: {e}")
        
        else:
            st.info("👆 Choose an input method to get started")
            st.markdown("""
            <div class="info-box">
            <strong>Ready to analyze handwritten digits:</strong><br><br>
            • **Real-time drawing analysis** - Draw directly in the canvas<br>
            • **Image upload processing** - Upload existing images<br>
            • **Sample digit testing** - Generate and test sample digits<br>
            • **Confidence scoring** - See how confident the model is<br>
            • **Probability distribution** - View all possible predictions<br>
            • **Download functionality** - Save your drawings<br>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick start guide
            with st.expander("🚀 Quick Start Guide"):
                st.markdown("""
                **For Best Results:**
                
                **1. Drawing Tab:**
                - Draw clearly in the center
                - Use thick, white strokes on black background
                - Make the digit fill most of the canvas
                - Avoid faint or blurry lines
                
                **2. Upload Tab:**
                - Use images with good contrast
                - White digits on black background work best
                - Clear, centered digits give best results
                
                **3. Sample Tab:**
                - Test with pre-generated digits
                - Verify model accuracy
                - Use as reference for your drawings
                """)

# Footer with project information
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <h3>🔍 Digital Vision AI - Interactive Digit Recognition</h3>
    <p><strong>Technologies Used:</strong> TensorFlow • Streamlit • Computer Vision • Machine Learning</p>
    <p><small>Includes: Real-time Drawing Analysis • Image Upload • Sample Testing • Confidence Scoring</small></p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()