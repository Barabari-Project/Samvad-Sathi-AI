from dotenv import load_dotenv
from phonemizer.punctuation import Punctuation
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from sequence_align.pairwise import needleman_wunsch


load_dotenv()

from openai import OpenAI

client = OpenAI()

def transcribe_audio(path):
    audio_file= open(path, "rb")

    transcription = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file,
        prompt="matlab, jaise ki, vagera-vagera, I'm like,you know what I mean, kind of, um, ah, huh, and so, so um, uh, and um, like um, so like, like it's, it's like, i mean, yeah, ok so, uh so, so uh, yeah so, you know, it's uh, uh and, and uh, like, kind",
    )

    return transcription.text

def generate_reference_phoneme(reference_text):
    text = Punctuation(';:,.!"?()').remove(reference_text)
    ref_words = [w.lower() for w in text.strip().split(' ') if w]
    
    backend = EspeakBackend('en-us')
    separator = Separator(phone='', word=None)
    lexicon = [ (word, backend.phonemize([word], separator=separator, strip=True)[0])
        for word in ref_words]
    
    return lexicon, ref_words 

def expected_phonetics(path,transcript=None):
    if not transcript:
        transcript = transcribe_audio(path=path)
    
    ref_phonetics = generate_reference_phoneme(transcript)
    return transcript,ref_phonetics
    