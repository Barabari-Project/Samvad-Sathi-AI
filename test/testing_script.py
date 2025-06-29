import pandas as pd
import requests
import gdown
import os
from tqdm import tqdm

# Paths and URLs
CSV_PATH = "Test_Cases.csv"
AUDIO_DIR = "audios"
os.makedirs(AUDIO_DIR, exist_ok=True)

# API Endpoints
URL1 = "http://localhost:8000/transcribe_whisper"
URL2 = "http://localhost:8000/pace-analysis"
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
        print(f"[INFO] Downloading from Google Drive: {file_id}")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", filename, quiet=False)
        return os.path.exists(filename)
    return False

# Process each row
for i, row in tqdm(df.iterrows(), total=len(df)):
    url = row.get("Drive link (male)")
    if pd.isna(url) or url.strip() == "":
        print(f"[SKIP] Row {i}: No drive link found.")
        continue

    audio_path = os.path.join(AUDIO_DIR, f"audio_{i}.wav")

    print(f"\n[PROCESSING] Row {i} - Downloading Audio...")
    if not download_audio(url, audio_path):
        print(f"[ERROR] Failed to download audio for row {i}")
        df.at[i, "transcript"] = "Download failed"
        continue

    try:
        # Step 1: Transcription
        print(f"[INFO] Transcribing audio for row {i}...")
        with open(audio_path, "rb") as f:
            r1 = requests.post(URL1, files={"file": f})
        r1.raise_for_status()
        response_json = r1.json()
        transcript = response_json["text"]
        df.at[i, "transcript"] = transcript
        print(f"[SUCCESS] Transcript: {transcript}")

        # Prepare payload
        payload = {"profile": response_json}

        # Step 2: Pacing
        print(f"[INFO] Getting pacing for row {i}...")
        r2 = requests.post(URL2, json=payload)
        r2.raise_for_status()
        res = r2.json()
        pacing_result = res["feedback"]
        df.at[i, "pacing"] = pacing_result
        print(f"[SUCCESS] Pacing: {pacing_result}")

        # Step 3: Pauses
        print(f"[INFO] Getting pauses for row {i}...")
        r3 = requests.post(URL3, json=payload)
        r3.raise_for_status()
        pauses_result = r3.json()
        df.at[i, "pauses"] = pauses_result
        print(f"[SUCCESS] Pauses: {pauses_result}")

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"[ERROR] Row {i}: {error_msg}")
        df.at[i, "transcript"] = error_msg

# Save final CSV
output_csv = "output_with_transcript_pacing_pauses.csv"
df.to_csv(output_csv, index=False)
print(f"\n[SAVED] Output written to: {output_csv}")
