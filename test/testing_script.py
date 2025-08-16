import pandas as pd
import requests
import gdown
import os
import concurrent.futures
from tqdm import tqdm
import json

# Paths and URLs
CSV_PATH = "Test_Cases_2.csv"
AUDIO_DIR = "audios"
os.makedirs(AUDIO_DIR, exist_ok=True)

# API Endpoints
URL1 = "http://localhost:8000/transcribe_whisper"
URL2 = "http://localhost:8000/pace-analysis"
URL3 = "http://localhost:8000/pauses-analysis"
URL4 = "http://localhost:8000/communication-based-analysis"
URL5 = "http://localhost:8000/transcribe_whisperx"

# Load CSV
df = pd.read_csv(CSV_PATH)

# Add new columns
df["transcript"] = ""
df["pacing"] = ""
df["pauses"] = ""
df["comunication-based-analysis"] = ""

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

# Parallel step execution
def process_analysis_requests(payload, transcript):
    results = {}
    def post_request(name, url, json_payload):
        try:
            r = requests.post(url, json=json_payload)
            r.raise_for_status()
            return name, r.json()
        except Exception as e:
            return name, f"Error: {str(e)}"

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(post_request, "pacing", URL2, {"words_timestamp": payload}),
            executor.submit(post_request, "pauses", URL3, {"words_timestamp": payload}),
            executor.submit(post_request, "communication", URL4, transcript)
        ]
        for future in concurrent.futures.as_completed(futures):
            name, result = future.result()
            results[name] = result

    return results

# Process each row
for i, row in tqdm(df.iterrows(), total=len(df)):
    url = row.get("Drive link (male)")
    if pd.isna(url) or url.strip() == "" : #  [82,41,33,30]
        print(f"[SKIP] Row {i}: No drive link found.")
        continue
    
    # if i not in [32,33,34,43]:
    #     continue
    

    try:
        
        # step 1: transcibe using whisperx
        print(f"[INFO] Transcribing audio for row {i}...")
        file_id = get_gdrive_file_id(url)
        donwload_url = f"https://drive.google.com/uc?id={file_id}"
        r1 = requests.post(URL5,json=donwload_url)
        r1.raise_for_status()
        response_json = r1.json()
        

        
        with open("r1_res.json",'w') as f:
            json.dump(response_json,f,indent=4)
            
        transcript = response_json["text"]
        
        df.at[i, "transcript"] = transcript
        print(f"[SUCCESS] Transcript: {transcript}")

        # Step 2-4: Run parallel
        print(f"[INFO] Running pacing, pauses, and communication analysis in parallel for row {i}...")
        # payload = {"profile": response_json}
        
        results = process_analysis_requests(response_json, transcript)

        try:
            df.at[i, "pacing"] = results.get("pacing", {}).get("feedback", "Error or missing feedback")
        except Exception as e:
            print("Error at decoding pacing")
            
        try:
            df.at[i, "pauses"] = results.get("pauses", {})
        except Exception as e:
            print("Error at decoding pauses")
            
        try:
            df.at[i, "comunication-based-analysis"] = results.get("communication", {})
        except Exception as e:
            print("Error at decoding comm")

        print(f"[SUCCESS] Pacing: {df.at[i, 'pacing']}")
        print(f"[SUCCESS] Pauses: {df.at[i, 'pauses']}")
        print(f"[SUCCESS] Communication: {df.at[i, 'comunication-based-analysis']}")
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"[ERROR] Row {i}: {error_msg}")
        df.at[i, "transcript"] = error_msg

# Save final CSV
output_csv = "Test_Cases_2.csv"
df.to_csv(output_csv, index=False)
print(f"\n[SAVED] Output written to: {output_csv}")