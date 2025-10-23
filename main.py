"""
Digital Vision AI - Complete Machine Learning Pipeline
Handwritten Digit Recognition, Iris Classification, and NLP Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import os
import warnings
import random
from datetime import datetime
import json

# Configure for optimal performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

# TensorFlow availability check
try:
    import tensorflow as tf
    # Only set GPU config if tensorflow is available and GPU is being used
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
    else:
        tf.config.set_visible_devices([], 'GPU')
    TENSORFLOW_AVAILABLE = True
    print(f"✅ TensorFlow {tf.__version__} loaded successfully")
except ImportError as e:
    print(f"❌ TensorFlow not available: {e}")
    print("💡 Install with: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False

# spaCy availability check
try:
    import spacy
    from spacy.matcher import PhraseMatcher
    SPACY_AVAILABLE = True
except ImportError:
    print("❌ spaCy not available")
    print("💡 Install with: pip install spacy")
    SPACY_AVAILABLE = False

# Create project directories
os.makedirs('output/iris_analysis', exist_ok=True)
os.makedirs('output/digit_recognition', exist_ok=True)
os.makedirs('output/nlp_analysis', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("🚀 Digital Vision AI - Starting Complete ML Pipeline...")
print("="*60)
print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# MODULE 1: IRIS CLASSIFICATION
# =============================================================================
print("\n" + "="*60)
print("MODULE 1: Iris Species Classification")
print("="*60)

def iris_classification():
    """Automated iris flower species classification with comprehensive analysis"""
    print("📊 Loading and preprocessing Iris dataset...")
    
    try:
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target
        target_names = iris.target_names
        
        print(f"📈 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"🎯 Classes: {', '.join(target_names)}")
        
        # Data quality assessment
        missing_values = X.isna().sum()
        if missing_values.sum() > 0:
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
            print("🔄 Missing values handled using mean imputation")
        else:
            print("✅ Data quality: No missing values found")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Data split: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
        
        # Model training
        print("🤖 Training Decision Tree classifier...")
        clf = DecisionTreeClassifier(max_depth=3, random_state=42)
        clf.fit(X_train, y_train)
        print("✅ Model training completed")
        
        # Model evaluation
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        
        print("\n📊 Performance Metrics:")
        print(f"   ✅ Accuracy:  {accuracy:.4f}")
        print(f"   ✅ Precision: {precision:.4f}")
        print(f"   ✅ Recall:    {recall:.4f}")
        
        # Feature importance analysis
        print("\n🔍 Feature Importance Analysis:")
        feature_importance = list(zip(iris.feature_names, clf.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for feature, importance in feature_importance:
            print(f"   {feature:20}: {importance:.4f}")
        
        # Comprehensive visualization
        plt.figure(figsize=(18, 6))
        
        # Decision Tree visualization
        plt.subplot(1, 3, 1)
        plot_tree(clf, feature_names=iris.feature_names, class_names=target_names, 
                  filled=True, rounded=True, fontsize=9)
        plt.title("Decision Tree Model", fontsize=12, pad=20)
        
        # Feature importance
        plt.subplot(1, 3, 2)
        features = [fi[0] for fi in feature_importance]
        importances = [fi[1] for fi in feature_importance]
        y_pos = np.arange(len(features))
        plt.barh(y_pos, importances, color='steelblue', alpha=0.8)
        plt.yticks(y_pos, features)
        plt.xlabel('Importance Score')
        plt.title('Feature Importance Ranking', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        
        # Confusion Matrix
        plt.subplot(1, 3, 3)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, 
                    yticklabels=target_names)
        plt.title('Confusion Matrix', fontsize=12)
        plt.ylabel('Actual Species')
        plt.xlabel('Predicted Species')
        
        plt.tight_layout()
        plt.savefig('output/iris_analysis/iris_classification_analysis.png', 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Save model performance
        performance_data = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'feature_importance': {feature: float(imp) for feature, imp in feature_importance},
            'test_samples': int(len(y_test))
        }
        
        with open('output/iris_analysis/performance.json', 'w') as f:
            json.dump(performance_data, f, indent=2)
        
        print("💾 Analysis saved to: output/iris_analysis/")
        return accuracy, clf
        
    except Exception as e:
        print(f"❌ Error in iris classification: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, None

# =============================================================================
# MODULE 2: HANDWRITTEN DIGIT RECOGNITION - IMPROVED FOR ACCURACY
# =============================================================================
print("\n" + "="*60)
print("MODULE 2: Handwritten Digit Recognition (CNN)")
print("="*60)

def digit_recognition():
    """Convolutional Neural Network for MNIST digit recognition with improved accuracy"""
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow required for digit recognition")
        print("💡 Please install TensorFlow: pip install tensorflow")
        return 0.0, None
    
    try:
        print("📊 Loading MNIST digit dataset...")
        
        # Load MNIST dataset using TensorFlow
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        print(f"📁 Training samples: {x_train.shape[0]:,}")
        print(f"📁 Test samples:     {x_test.shape[0]:,}")
        print(f"📏 Image dimensions: {x_train.shape[1:]}")
        
        # Enhanced Data preprocessing pipeline
        print("🔄 Preprocessing data for optimal model performance...")
        
        # Normalize pixel values to 0-1 range
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        
        # Add channel dimension
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
        
        print("✅ Data preprocessing completed")
        
        # Enhanced CNN Model Architecture for better accuracy
        def build_improved_cnn_model():
            model = tf.keras.Sequential([
                # First convolutional block with batch normalization
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                # Second convolutional block
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                # Third convolutional block
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.25),
                
                # Classification layers
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            
            # Use Adam optimizer with custom learning rate
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            return model
        
        model = build_improved_cnn_model()
        
        print("\n🧠 Enhanced Neural Network Architecture:")
        print("   Convolutional Layers: 3 blocks with batch normalization")
        print("   Pooling Layers: 2")
        print("   Dense Layers: 3")
        print("   Regularization: Dropout and BatchNorm")
        print("   Total Parameters: {:,}".format(model.count_params()))
        
        # Enhanced callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        # Data augmentation for better generalization
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1
        )
        
        # Model training with data augmentation
        print("\n🎯 Training enhanced neural network...")
        print("   Target: Achieve >99% test accuracy")
        
        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=128),
            epochs=30,
            validation_data=(x_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Model evaluation
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"\n✅ Model Evaluation:")
        print(f"   Test Accuracy: {test_acc:.4f}")
        print(f"   Test Loss:     {test_loss:.4f}")
        
        if test_acc > 0.99:
            print("   🎯 EXCELLENT: Accuracy > 99%!")
        elif test_acc > 0.98:
            print("   🎯 VERY GOOD: Accuracy > 98%!")
        elif test_acc > 0.95:
            print("   ✅ GOOD: Accuracy > 95%!")
        else:
            print("   ⚠️  Needs improvement: Accuracy < 95%")
        
        # Enhanced visualization
        plt.figure(figsize=(20, 6))
        
        # Training history - Loss
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Training Loss', linewidth=2, color='#e74c3c')
        plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#3498db')
        plt.title('Model Loss Progress', fontsize=12)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Training history - Accuracy
        plt.subplot(1, 3, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='#2ecc71')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#f39c12')
        plt.title('Model Accuracy Progress', fontsize=12)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Sample predictions visualization with confidence scores
        plt.subplot(1, 3, 3)
        indices = random.sample(range(len(x_test)), 12)
        sample_images = x_test[indices]
        sample_labels = y_test[indices]
        predictions = model.predict(sample_images, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        for i, idx in enumerate(range(12)):
            plt.subplot(3, 4, i+1)
            plt.imshow(sample_images[i].squeeze(), cmap='gray')
            true_label = sample_labels[i]
            pred_label = predicted_labels[i]
            confidence = confidences[i]
            color = 'green' if true_label == pred_label else 'red'
            plt.title(f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}", 
                     color=color, fontsize=7)
            plt.axis('off')
        
        plt.suptitle('Enhanced Sample Digit Predictions', fontsize=14)
        plt.tight_layout()
        plt.savefig('output/digit_recognition/digit_recognition_analysis.png', 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Detailed confidence analysis
        plt.figure(figsize=(15, 5))
        
        # Confidence distribution
        plt.subplot(1, 3, 1)
        all_predictions = model.predict(x_test, verbose=0)
        all_confidences = np.max(all_predictions, axis=1)
        all_predicted = np.argmax(all_predictions, axis=1)
        
        correct_confidences = all_confidences[all_predicted == y_test]
        incorrect_confidences = all_confidences[all_predicted != y_test]
        
        plt.hist(correct_confidences, bins=30, alpha=0.7, color='#2ecc71', label='Correct', edgecolor='black')
        plt.hist(incorrect_confidences, bins=30, alpha=0.7, color='#e74c3c', label='Incorrect', edgecolor='black')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution: Correct vs Incorrect')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Per-digit accuracy
        plt.subplot(1, 3, 2)
        digit_accuracy = []
        for digit in range(10):
            digit_mask = y_test == digit
            digit_accuracy.append(np.mean(all_predicted[digit_mask] == y_test[digit_mask]))
        
        plt.bar(range(10), digit_accuracy, color='#3498db', alpha=0.8)
        plt.xlabel('Digit')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per Digit')
        plt.xticks(range(10))
        plt.grid(axis='y', alpha=0.3)
        
        # Confusion matrix
        plt.subplot(1, 3, 3)
        cm = confusion_matrix(y_test, all_predicted)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('output/digit_recognition/detailed_analysis.png', 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Save trained model
        model.save('models/digit_recognition_model.h5')
        print("💾 Model saved to: models/digit_recognition_model.h5")
        
        # Enhanced training history with per-digit metrics
        history_data = {
            'final_accuracy': float(test_acc),
            'final_loss': float(test_loss),
            'digit_accuracy': {str(i): float(acc) for i, acc in enumerate(digit_accuracy)},
            'training_history': {
                'loss': [float(x) for x in history.history['loss']],
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']]
            },
            'model_architecture': {
                'conv_layers': 5,
                'dense_layers': 3,
                'parameters': int(model.count_params()),
                'optimizer': 'Adam',
                'learning_rate': 0.001
            }
        }
        
        with open('output/digit_recognition/training_history.json', 'w') as f:
            json.dump(history_data, f, indent=2)
        
        # Print detailed performance analysis
        print(f"\n📊 Detailed Performance Analysis:")
        print(f"   Average Confidence (Correct): {np.mean(correct_confidences):.3f}")
        print(f"   Average Confidence (Incorrect): {np.mean(incorrect_confidences):.3f}")
        print(f"   Most Accurate Digit: {np.argmax(digit_accuracy)} ({np.max(digit_accuracy):.3f})")
        print(f"   Least Accurate Digit: {np.argmin(digit_accuracy)} ({np.min(digit_accuracy):.3f})")
        
        return test_acc, model
        
    except Exception as e:
        print(f"❌ Error in digit recognition: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, None

# =============================================================================
# MODULE 3: TEXT ANALYSIS & SENTIMENT
# =============================================================================
print("\n" + "="*60)
print("MODULE 3: Text Analysis & Sentiment Detection")
print("="*60)

def text_analysis():
    """Natural Language Processing for text analysis and sentiment detection"""
    if not SPACY_AVAILABLE:
        print("❌ spaCy required for advanced text analysis")
        return basic_text_analysis()
    
    try:
        print("📝 Initializing text analysis pipeline...")
        
        # Load spaCy model
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✅ NLP model loaded successfully")
        except OSError:
            print("❌ spaCy English model not found")
            print("💡 Install with: python -m spacy download en_core_web_sm")
            return basic_text_analysis()
        
        # Enhanced product reviews dataset
        reviews = [
            "I absolutely love my new iPhone 14 Pro from Apple. The camera quality is amazing and battery life lasts all day!",
            "This Samsung Galaxy S23 is terrible. The screen cracked after one week and customer service was unhelpful.",
            "My Dell XPS laptop from Amazon works perfectly for programming and gaming. Highly recommended!",
            "The Sony headphones I bought have poor sound quality and the build feels cheap. Very disappointed.",
            "Google Pixel 7 has an incredible camera and clean Android experience. Best phone I've ever owned!",
            "This HP printer constantly jams and the ink is too expensive. Would not buy again.",
            "Microsoft Surface Pro is fantastic for work and travel. The pen input is very responsive.",
            "The Lenovo ThinkPad keyboard is comfortable but the battery life is shorter than expected.",
            "Amazing product from Apple! The build quality is exceptional and performance is outstanding.",
            "Poor customer service from Samsung. The device stopped working after 2 months."
        ]
        
        print(f"📊 Processing {len(reviews)} product reviews...")
        
        # Initialize pattern matcher for product recognition
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        product_terms = [
            "iphone", "samsung galaxy", "dell xps", "sony headphones", 
            "google pixel", "hp printer", "microsoft surface", "lenovo thinkpad",
            "apple", "samsung", "dell", "sony", "google", "hp", "microsoft", "lenovo"
        ]
        patterns = [nlp(text) for text in product_terms]
        matcher.add("PRODUCT_BRAND", patterns)
        
        # Enhanced sentiment analysis lexicon
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
        
        def analyze_sentiment(text):
            """Advanced sentiment analysis with word counting"""
            text_low = text.lower()
            words = text_low.split()
            pos = sum(1 for word in words if word in positive_words)
            neg = sum(1 for word in words if word in negative_words)
            
            # Calculate sentiment score
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
        
        # Process all reviews
        print("\n🔍 Analyzing text data...")
        
        sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        all_entities = []
        analysis_results = []
        
        for i, review_text in enumerate(reviews, 1):
            doc = nlp(review_text)
            
            # Named Entity Recognition
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            all_entities.extend(entities)
            
            # Product/Brand recognition
            matches = matcher(doc)
            found_products = {doc[start:end].text for match_id, start, end in matches}
            
            # Sentiment analysis
            sentiment, pos_count, neg_count, sentiment_score = analyze_sentiment(review_text)
            sentiment_counts[sentiment] += 1
            
            analysis_results.append({
                'review_id': i,
                'review_text': review_text,
                'entities': entities,
                'products': list(found_products),
                'sentiment': sentiment,
                'positive_words': pos_count,
                'negative_words': neg_count,
                'sentiment_score': float(sentiment_score)
            })
            
            print(f"\n📝 Review {i}:")
            print(f"   Products: {list(found_products)}")
            print(f"   Sentiment: {sentiment} (Score: {sentiment_score:.2f})")
            print(f"   Positive words: {pos_count}, Negative words: {neg_count}")
        
        # Analysis summary
        print(f"\n📈 Text Analysis Summary:")
        print(f"   Documents processed: {len(reviews)}")
        print(f"   Entities extracted:  {len(all_entities)}")
        print(f"   Sentiment distribution: {sentiment_counts}")
        
        # Entity analysis
        entity_types = {}
        for entity, etype in all_entities:
            entity_types[etype] = entity_types.get(etype, 0) + 1
        
        print(f"   Entity types: {entity_types}")
        
        # Visualization
        plt.figure(figsize=(15, 5))
        
        # Sentiment distribution
        plt.subplot(1, 3, 1)
        sentiments = list(sentiment_counts.keys())
        counts = list(sentiment_counts.values())
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        bars = plt.bar(sentiments, counts, color=colors, alpha=0.8)
        plt.title('Sentiment Analysis', fontsize=12)
        plt.ylabel('Number of Reviews')
        
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Entity distribution
        plt.subplot(1, 3, 2)
        if entity_types:
            labels = list(entity_types.keys())
            values = list(entity_types.values())
            plt.bar(labels, values, color='#3498db', alpha=0.8)
            plt.title('Entity Type Distribution', fontsize=12)
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
        
        # Word sentiment comparison
        plt.subplot(1, 3, 3)
        pos_total = sum(r['positive_words'] for r in analysis_results)
        neg_total = sum(r['negative_words'] for r in analysis_results)
        plt.bar(['Positive Words', 'Negative Words'], [pos_total, neg_total], 
                color=['#2ecc71', '#e74c3c'], alpha=0.8)
        plt.title('Sentiment Word Frequency', fontsize=12)
        plt.ylabel('Total Occurrences')
        
        plt.suptitle('Advanced Text Analysis of Product Reviews', fontsize=14)
        plt.tight_layout()
        plt.savefig('output/nlp_analysis/text_analysis_results.png', 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Save analysis results
        with open('output/nlp_analysis/analysis_results.json', 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        summary_data = {
            'total_reviews': len(reviews),
            'total_entities': len(all_entities),
            'sentiment_distribution': sentiment_counts,
            'entity_types': entity_types,
            'word_frequency': {
                'positive_words': pos_total,
                'negative_words': neg_total
            }
        }
        
        with open('output/nlp_analysis/summary.json', 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print("💾 Analysis saved to: output/nlp_analysis/")
        return len(reviews), sentiment_counts, analysis_results
        
    except Exception as e:
        print(f"❌ Error in text analysis: {e}")
        import traceback
        traceback.print_exc()
        return basic_text_analysis()

def basic_text_analysis():
    """Fallback text analysis without spaCy"""
    print("🔄 Using basic text analysis (spaCy not available)")
    
    reviews = [
        "I love the Acme Turbo Blender. The Acme brand is reliable.",
        "The Zerex headphones sound tinny. Not worth the money.",
        "Bought the UltraCup by CupCo — great insulation and design.",
        "Product: FastCups mug. Brand: FastGoods. Battery life is poor."
    ]
    
    positive_words = {"love", "great", "excellent", "good", "reliable", "perfect"}
    negative_words = {"poor", "bad", "terrible", "not worth", "hate", "tinny"}
    
    sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    
    for i, review in enumerate(reviews, 1):
        text_low = review.lower()
        pos = sum(1 for word in text_low.split() if word in positive_words)
        neg = sum(1 for word in text_low.split() if word in negative_words)
        
        if pos > neg:
            sentiment = "Positive"
        elif neg > pos:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        sentiment_counts[sentiment] += 1
        print(f"Review {i}: {sentiment} (+{pos}, -{neg})")
    
    print(f"\n📊 Basic Analysis: {len(reviews)} reviews processed")
    print(f"   Sentiment: {sentiment_counts}")
    
    return len(reviews), sentiment_counts, []

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("🚀 Digital Vision AI - Machine Learning Pipeline")
    print("="*60)
    
    # Track execution results
    results = {}
    
    try:
        # Execute all modules
        print("\n" + "="*60)
        print("EXECUTING: Iris Classification")
        print("="*60)
        iris_accuracy, iris_model = iris_classification()
        results['iris'] = {'accuracy': iris_accuracy, 'success': iris_model is not None}
        
        print("\n" + "="*60)
        print("EXECUTING: Digit Recognition")
        print("="*60)
        digit_accuracy, digit_model = digit_recognition()
        results['digit'] = {'accuracy': digit_accuracy, 'success': digit_model is not None}
        
        print("\n" + "="*60)
        print("EXECUTING: Text Analysis")
        print("="*60)
        reviews_count, sentiment_stats, text_results = text_analysis()
        results['text'] = {'reviews_processed': reviews_count, 'success': True}
        
    except Exception as e:
        print(f"❌ Pipeline execution error: {e}")
        import traceback
        traceback.print_exc()
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 PIPELINE EXECUTION COMPLETE")
    print("="*60)
    
    print(f"📊 Module Performance Summary:")
    if results.get('iris', {}).get('success'):
        accuracy = results['iris']['accuracy']
        if accuracy > 0.95:
            status = "(>95% 🎯 EXCELLENT)"
        elif accuracy > 0.90:
            status = "(>90% 👍 VERY GOOD)"
        else:
            status = "(<90% ✅ GOOD)"
        print(f"   ✅ Iris Classification:    {accuracy:.4f} accuracy {status}")
    else:
        print(f"   ❌ Iris Classification:    Failed")
    
    if results.get('digit', {}).get('success'):
        accuracy = results['digit']['accuracy']
        if accuracy > 0.99:
            status = "(>99% 🎯 EXCELLENT)"
        elif accuracy > 0.98:
            status = "(>98% 🎯 VERY GOOD)"
        elif accuracy > 0.95:
            status = "(>95% ✅ GOOD)"
        else:
            status = "(<95% ⚠️ NEEDS IMPROVEMENT)"
        print(f"   ✅ Digit Recognition:      {accuracy:.4f} accuracy {status}")
    else:
        print(f"   ❌ Digit Recognition:      TensorFlow not available")
    
    if results.get('text', {}).get('success'):
        print(f"   ✅ Text Analysis:          {results['text']['reviews_processed']} documents processed")
    else:
        print(f"   ❌ Text Analysis:          Failed")
    
    print(f"\n💾 Output Files Generated:")
    print(f"   Models:          models/digit_recognition_model.h5")
    print(f"   Visualizations:  output/iris_analysis/")
    print(f"                   output/digit_recognition/")
    print(f"                   output/nlp_analysis/")
    print(f"   Data:           Various JSON files with analysis results")
    
    print(f"\n🚀 Next Steps:")
    print(f"   To launch the interactive app: streamlit run app.py")
    print(f"   The app provides real-time digit recognition with drawing interface")
    
    print(f"\n🕒 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)