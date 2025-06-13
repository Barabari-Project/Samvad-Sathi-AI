# First, you need to install the necessary libraries.
# You can do this by running the following command in your terminal:
# pip install opensmile-python pandas

import opensmile
import pandas as pd
import os

def extract_paralinguistic_features(audio_file_path):
    """
    Extracts paralinguistic features from an audio file using the openSMILE toolkit.

    This function uses the eGeMAPS (extended Geneva Minimalistic Acoustic Parameter Set),
    which is well-suited for analyzing features related to emotion, prosody, and voice quality.

    Args:
        audio_file_path (str): The full path to the audio file (e.g., 'C:/audio/sample.wav').

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted features.
                          Returns None if the file is not found.
    """
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found at '{audio_file_path}'")
        return None

    # 1. Create a Smile object
    # This object represents the openSMILE engine.
    # We specify the eGeMAPS feature set, which contains 88 high-level acoustic features.
    # We use FeatureLevel.Functionals to get statistical summaries (like mean, stddev)
    # of low-level descriptors over the entire audio file. These are the "paralinguistic features".
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPS,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    print(f"Processing file: {audio_file_path}...")

    # 2. Extract features from the audio file
    # The process_file() method runs the openSMILE executable on the audio file
    # and returns the results as a pandas DataFrame.
    try:
        features = smile.process_file(audio_file_path)
        print("Feature extraction complete.")
        return features
    except Exception as e:
        print(f"An error occurred during feature extraction: {e}")
        return None

def map_features_to_concepts(feature_df):
    """
    Explains how the extracted features relate to high-level paralinguistic concepts.
    """
    if feature_df is None:
        return

    print("\n--- Mapping Extracted Features to Paralinguistic Concepts ---")

    print("\n1. Speech Rate, Pauses & Hesitations:")
    print("   - These are captured by features describing the timing and voiced/unvoiced segments.")
    print("   - Look for columns like: `VoicedSegmentsPerSecond`, `MeanVoicedSegmentLength`, `MeanUnvoicedSegmentLength`, `nsp_not_sil_rate` (rate of non-silence period).")

    print("\n2. Prosody (Pitch and Intonation):")
    print("   - This is primarily described by statistics of the fundamental frequency (F0).")
    print("   - Look for columns starting with `F0semitoneFrom27.5Hz_...`, such as `..._mean`, `..._stddevNorm` (pitch variability), and `..._slope`.")

    print("\n3. Volume Dynamics (Loudness):")
    print("   - This is captured by energy-related features.")
    print("   - Look for columns starting with `loudness_...` and `HNR...` (Harmonics-to-Noise Ratio), like `loudness_sma3_stddevNorm` (loudness variation).")

    print("\n4. Emotional Tone (Voice Quality & Timbre):")
    print("   - This is a complex concept derived from a combination of features, including:")
    print("     - Jitter & Shimmer: Measures of frequency and amplitude instability (e.g., `jitterLocal_sma3nz_stddevNorm`, `shimmerLoc_sma3nz_stddevNorm`).")
    print("     - Spectral Balance: Distribution of energy across different frequencies (e.g., `spectralFlux_sma3_stddevNorm`, `alphaRatio_sma3_amean`).")
    print("     - All of the above features (Prosody, Volume) also contribute heavily to the machine learning models that predict emotional tone.")
    print("----------------------------------------------------------\n")


# --- Main Execution ---
if __name__ == "__main__":
    # IMPORTANT: Replace this with the actual path to your audio file.
    # The file should be in a standard format like .wav, .mp3, .flac, etc.
    # For best results, use a .wav file with a 16kHz sample rate.
    audio_file = "harvard.wav" # <--- CHANGE THIS

    # Create a dummy audio file for demonstration if it doesn't exist.
    if audio_file == "your_audio_file.wav" and not os.path.exists(audio_file):
        print("Creating a dummy sine wave audio file for demonstration: 'your_audio_file.wav'")
        print("Please replace this with your actual audio file.")
        try:
            import numpy as np
            import scipy.io.wavfile
            sample_rate = 16000
            duration = 5
            frequency = 440
            t = np.linspace(0., duration, int(sample_rate * duration))
            amplitude = np.iinfo(np.int16).max * 0.5
            # Add some pauses
            t[int(1*sample_rate):int(1.5*sample_rate)] = 0
            t[int(3*sample_rate):int(3.8*sample_rate)] = 0
            data = amplitude * np.sin(2. * np.pi * frequency * t)
            scipy.io.wavfile.write(audio_file, sample_rate, data.astype(np.int16))
        except ImportError:
            print("\nCould not create a dummy audio file because 'numpy' or 'scipy' is not installed.")
            print("Please run: pip install numpy scipy")
            print("And then create a file named 'your_audio_file.wav' manually.")


    # Extract the features
    extracted_features = extract_paralinguistic_features(audio_file)

    if extracted_features is not None:
        # Display the results
        # We transpose (.T) the DataFrame for better readability in the console.
        print("\n--- Extracted Features (eGeMAPS set) ---")
        pd.set_option('display.max_rows', None)
        print(extracted_features.T)

        # Explain the features
        map_features_to_concepts(extracted_features)
