# mood_classifier.py
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
from audio_processing import extract_features

def load_data(data_dir='dataset/'):
    """
    Loads audio data and extracts features.
    Args:
        data_dir (str): Directory containing labeled audio files.
    Returns:
        np.array, np.array: Features and corresponding labels.
    """
    features, labels = [], []
    # Iterate through each file in the dataset folder
    for filename in os.listdir(data_dir):
        if filename.endswith('.wav'):
            label = filename.split('_')[0]  # Extract label based on filename, e.g., 'energetic_1.wav' -> 'energetic'
            file_path = os.path.join(data_dir, filename)
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(label)
    return np.array(features), np.array(labels)

def train_model():
    """
    Train a mood classification model and save it as a pickle file.
    """
    # Load data
    X, y = load_data()
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")

    # Save the trained model and label encoder
    with open('mood_classifier.pkl', 'wb') as f:
        pickle.dump((model, label_encoder), f)
    print("Model saved as mood_classifier.pkl")

if __name__ == "__main__":
    train_model()
