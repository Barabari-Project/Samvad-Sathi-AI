from pydub import AudioSegment
import os

def convert_to_wav_16k(input_file, output_file=None):
    """
    Convert any audio file to .wav format with 16kHz sample rate.
    
    Args:
        input_file (str): Path to the input audio file.
        output_file (str, optional): Path to save the output .wav file.
                                     If None, it will use same name as input with .wav extension.
    Returns:
        str: Path to the converted .wav file.
    """
    if output_file is None:
        base = os.path.splitext(input_file)[0]
        output_file = base + ".wav"

    # Load audio
    audio = AudioSegment.from_file(input_file)

    # Set frame rate to 16 kHz
    audio = audio.set_frame_rate(16000).set_channels(1)  # mono

    # Export as .wav
    audio.export(output_file, format="wav")
    return output_file

# Example usage
if __name__ == "__main__":
    converted_path = convert_to_wav_16k("example.mp3")
    print("Converted file saved at:", converted_path)
