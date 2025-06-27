import os
from pydub import AudioSegment

def convert_mp3_directory_to_wav(input_dir, output_dir, target_sample_rate=16000):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.mp3'):
            mp3_path = os.path.join(input_dir, filename)
            wav_filename = os.path.splitext(filename)[0] + '.wav'
            wav_path = os.path.join(output_dir, wav_filename)

            # Load and convert audio
            audio = AudioSegment.from_mp3(mp3_path)
            audio = audio.set_frame_rate(target_sample_rate)
            audio.export(wav_path, format='wav')
            print(f"Converted: {mp3_path} -> {wav_path} ({target_sample_rate} Hz)")

# Example usage
input_directory = "dataset"
output_directory = "dataset"
convert_mp3_directory_to_wav(input_directory, output_directory)
