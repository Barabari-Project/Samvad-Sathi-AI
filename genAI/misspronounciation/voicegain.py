import requests, time, os, json, re
from dotenv import load_dotenv
load_dotenv()
JWT = os.getenv("VOICEGAIN_API_KEY")

def get_final_transcription_json(audio_path: str):
    
    platform = "voicegain"
    audio_type = "audio/wav"

    headers = {"Authorization": JWT}
    
    # Upload audio file
    data_url = f"https://api.{platform}.ai/v1/data/file"
    data_body = {
        "name": re.sub("[^A-Za-z0-9]+", "-", audio_path),
        "description": audio_path,
        "contentType": audio_type,
        "tags": ["test"]
    }

    multipart_form_data = {
        'file': (audio_path, open(audio_path, 'rb'), audio_type),
        'objectdata': (None, json.dumps(data_body), "application/json")
    }

    data_response = requests.post(data_url, files=multipart_form_data, headers=headers).json()
    object_id = data_response["objectId"]

    # Start ASR session
    asr_body = {
        "sessions": [
            {
                "asyncMode": "OFF-LINE",
                "poll": {
                    "afterlife": 60000,
                    "persist": 600000
                },
                "content": {
                    "incremental": ["progress"],
                    "full": ["transcript", "words"]
                }
            }
        ],
        "audio": {
            "source": {
                "dataStore": {
                    "uuid": object_id
                }
            }
        }
    }

    asr_url = f"https://api.{platform}.ai/v1/asr/transcribe/async"
    asr_response = requests.post(asr_url, json=asr_body, headers=headers).json()
    polling_url = asr_response["sessions"][0]["poll"]["url"]

    # Poll until final result is ready
    index = 0
    while True:
        time.sleep(0.3 if index < 5 else 4.9)
        poll_response = requests.get(polling_url + "?full=false", headers=headers).json()
        is_final = poll_response["result"]["final"]
        if is_final:
            break
        index += 1

    # Get full final result
    final_response = requests.get(polling_url + "?full=true", headers=headers).json()
    return final_response

# final_json = get_final_transcription_json("dataset/smit_pizza_true.wav")

import time
a = time.time()
final_json = get_final_transcription_json("audios/my_answer.wav")
a = time.time() - a
print(json.dumps(final_json, indent=2))
print(a)