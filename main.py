# main.py
import os
import pickle
from audio_processing import extract_features

# Load the trained model and label encoder
with open('mood_classifier.pkl', 'rb') as f:
    model, label_encoder = pickle.load(f)

# Music recommendations for each mood
mood_to_music = {
    'energetic': ['energetic_1.wav', 'energetic_2.wav', 'energetic_3.wav'],
    'cool': ['cool_1.wav', 'cool_2.wav', 'cool_3.wav'],
    'romantic': ['romantic_1.wav', 'romantic_2.wav', 'romantic_3.wav']
}

def predict_mood(file_path):
    """
    Predict the mood of an audio file.
    Args:
        file_path (str): Path to the audio file.
    Returns:
        str: Predicted mood label.
    """
    # Extract features from the input file
    features = extract_features(file_path)
    if features is not None:
        # Reshape features for prediction
        features = features.reshape(1, -1)
        mood_index = model.predict(features)[0]
        mood = label_encoder.inverse_transform([mood_index])[0]
        return mood
    return None

def recommend_music(mood):
    """
    Recommend music based on mood.
    Args:
        mood (str): Detected mood label.
    Returns:
        str: Recommended music file.
    """
    return mood_to_music.get(mood, ['default.wav'])[0]

# Example usage
if __name__ == "__main__":
    test_audio = 'dataset/energetic_1.wav'  # Replace with your test audio file path
    detected_mood = predict_mood(test_audio)
    if detected_mood:
        print(f"Detected Mood: {detected_mood}")
        recommended_song = recommend_music(detected_mood)
        print(f"Recommended Song: {recommended_song}")
    else:
        print("Could not detect mood.")
