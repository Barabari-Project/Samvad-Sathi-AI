import phonemizer
from phonemizer.punctuation import Punctuation
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
import Levenshtein
import re
import os
from dotenv import load_dotenv
from tabulate import tabulate


load_dotenv()

from openai import OpenAI

client = OpenAI()

def transcribe_audio(path):
    audio_file= open(path, "rb")

    transcription = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file,
        prompt="matlab, jaise ki, vagera-vagera, I'm like,you know what I mean, kind of, um, ah, huh, and so, so um, uh, and um, like um, so like, like it's, it's like, i mean, yeah, ok so, uh so, so uh, yeah so, you know, it's uh, uh and, and uh, like, kind",
        include='logprobs'
    )
    print(transcription)
    return transcription

transcribe_audio('audios/my_answer.mp3')