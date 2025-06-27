# modal_server.py

import base64
import tempfile
import traceback
from pydantic import BaseModel
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import modal

# === Modal App ===
app = modal.App("asr-phoneme-recognizer-v8")  # Updated name for fresh deployment
auth_scheme = HTTPBearer()

# === Modal Image Definition ===
whisper_image = (
    modal.Image.micromamba()
    .apt_install("ffmpeg", "espeak")
    .env({"ESPEAK_LIBRARY": "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1"})
    .micromamba_install(
        "cudatoolkit=11.8",
        "cudnn=8.1.0",
        "cuda-nvcc",
        channels=["conda-forge", "nvidia"],
    )
    .pip_install(
        "torch==2.0.1",
        "transformers==4.37.2",
        "phonemizer",
        "pydantic==1.10.12",
        "fastapi==0.100.0",
        "python-multipart",
        "requests",
        "uvicorn",
        "librosa",  # Add for better audio processing
        "soundfile"  # Add for audio file handling
    )
)


# === Request Model ===
class TranscriptionRequest(BaseModel):
    audio: str  # base64-encoded mp3 string

# === Model Class ===
@app.cls(
    gpu="T4",  # Fixed GPU specification
    min_containers=1,  # Replaced keep_warm
    scaledown_window=120,  # Replaced container_idle_timeout
    image=whisper_image,
    secrets=[
        modal.Secret.from_name("huggingface-secret-2"),
        modal.Secret.from_name("whisper-web-auth-token"),
    ],
)
class Model:
    initialized: bool = False  # Class-level attribute to avoid __init__
    
    def _initialize(self):
        if self.initialized:
            return
        print("🚀 Initializing ASR pipeline...")
        try:
            from transformers import pipeline, AutoTokenizer, AutoFeatureExtractor, Wav2Vec2Processor

            # Use the processor instead of separate tokenizer/feature_extractor
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

            # Initialize pipeline with proper padding configuration
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model="facebook/wav2vec2-xlsr-53-espeak-cv-ft",
                processor=processor,
                device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
            )
            self.initialized = True
            print("✅ Pipeline loaded and ready!")
        except Exception as e:
            print("❌ Failed to initialize pipeline")
            traceback.print_exc()
            raise RuntimeError("Model initialization failed") from e


    @modal.method()
    def transcribe_audio(self, audio_data: bytes) -> dict:
        self._initialize()  # Ensure model is loaded
        
        with tempfile.NamedTemporaryFile(suffix=".mp3") as temp_audio:
            temp_audio.write(audio_data)
            temp_audio.flush()
            
            try:
                # Use proper parameters for Wav2Vec2
                result = self.pipe(
                    temp_audio.name,
                    chunk_length_s=30,
                    stride_length_s=5,  # Add stride for better chunking
                    return_timestamps="word",
                    # Force padding for consistent tensor shapes
                    padding=True,
                    truncation=True
                )
                return result
            except Exception as e:
                print(f"Transcription error: {e}")
                # Fallback: try without chunking for shorter audio
                try:
                    result = self.pipe(temp_audio.name, return_timestamps="word")
                    return result
                except Exception as fallback_error:
                    raise RuntimeError(f"Transcription failed: {fallback_error}")


    @modal.fastapi_endpoint(method="POST")  # Updated to fastapi_endpoint
    async def transcribe(self, request: TranscriptionRequest):
        try:
            audio_data = base64.b64decode(request.audio.split(",")[1])
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid base64 audio encoding"
            )

        # Validate audio length
        if len(audio_data) < 1024:
            raise HTTPException(
                status_code=400,
                detail="Audio too short (min 1KB required)"
            )

        try:
            return await self.transcribe_audio.remote.aio(audio_data)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )

    @modal.fastapi_endpoint(method="GET")  # Updated to fastapi_endpoint
    def ping(self):
        return {"message": "Phoneme recognizer is up and running!"}