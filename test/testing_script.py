import pandas as pd
import requests
import gdown
import os
from tqdm import tqdm

# Paths and URLs
CSV_PATH = "Test_Cases.csv" # path to Samvad Sathi Test Cases
AUDIO_DIR = "audios"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Endpoint URLs
URL1 = "http://localhost:8000/transcribe_whisper"
URL2 = "http://localhost:8000/misspronounciation-analysis"
URL3 = "http://localhost:8000/pauses-analysis"

# Load CSV
df = pd.read_csv(CSV_PATH)

# Add new columns
df["transcript"] = ""
df["pacing"] = ""
df["pauses"] = ""

# Extract file ID from Google Drive link
def get_gdrive_file_id(url):
    if "id=" in url:
        return url.split("id=")[-1].split("&")[0]
    elif "file/d/" in url:
        return url.split("file/d/")[1].split("/")[0]
    return None

# Download from Google Drive using gdown
def download_audio(drive_url, filename):
    file_id = get_gdrive_file_id(drive_url)
    if file_id:
        gdown.download(f"https://drive.google.com/uc?id={file_id}", filename, quiet=False)
        return os.path.exists(filename)
    return False

# Process each row
for i, row in tqdm(df.iterrows(), total=len(df)):
    url = row.get("Drive link (male)")
    if pd.isna(url) or url.strip() == "":
        continue

    audio_path = os.path.join(AUDIO_DIR, f"audio_{i}.wav")

    if not download_audio(url, audio_path):
        df.at[i, "transcript"] = "Download failed"
        continue

    try:
        # URL1: Transcribe
        with open(audio_path, "rb") as f:
            r1 = requests.post(URL1, files={"file": f})
        r1.raise_for_status()
        transcript = r1.json()["words"]  # Assuming server returns: {"words": "text..."}
        df.at[i, "transcript"] = transcript

        # Payload to URL2 and URL3
        payload = {"profile": {"words": transcript}}

        # URL2: Pacing
        r2 = requests.post(URL2, json=payload)
        r2.raise_for_status()
        df.at[i, "pacing"] = r2.text.strip()

        # URL3: Pauses
        r3 = requests.post(URL3, json=payload)
        r3.raise_for_status()
        df.at[i, "pauses"] = r3.text.strip()

    except Exception as e:
        df.at[i, "transcript"] = f"Error: {e}"

# Save final output
df.to_csv("output_with_transcript_pacing_pauses.csv", index=False)
print("Saved: output_with_transcript_pacing_pauses.csv")