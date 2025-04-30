# audio_processing.py
import librosa 
import numpy as np 

def extract_features(file_path):
    """
    Extracts various audio features like MFCC, Chroma, Mel Spectrogram, and Spectral Contrast.
    Args:
        file_path (str): Path to the audio file.
    Returns:
        np.array: Concatenated feature array.
    """
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, duration=30)

        # Extract various features
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)

        # Concatenate features
        features = np.hstack([mfccs, chroma, mel, contrast])
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
