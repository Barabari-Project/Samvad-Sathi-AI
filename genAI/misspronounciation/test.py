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
    )

    return transcription

def generate_reference_phoneme(reference_text):
    text = Punctuation(';:,.!"?()').remove(reference_text)
    ref_words = [w.lower() for w in text.strip().split(' ') if w]
    
    backend = EspeakBackend('en-us')
    separator = Separator(phone='', word=None)
    lexicon = [ (word, backend.phonemize([word], separator=separator, strip=True)[0])
        for word in ref_words]
    
    return lexicon, ref_words 

# reference_text = '''of, yeah, so um, one technical challenge was like, ODIA language, you know, it had, like, way less data than others, so, model was like, kind of struggling to learn it properly, I'm like, you know, the loss was, and all, loss was high and all, and um, it had, just wasn't working good, so um, we tried to, like, fix that by making batches balanced, like, you know, make sure each batch had thoda-thoda from every language, so like, even if ODIA ka data kam tha, it still came in training, I mean, it helped, kind of, but not fully, cause, data hi kam tha, you know what I mean, and so, um, yeah, so, training the model itself was like, tough, it's like, our first time training something this big, memory ka kaafi issue de raha tha, and GPU be limited tha, so, um, we did, like, the gradient accumulation and stuff, and processed data in chunks, like, um, small parts, so it doesn't crash or something, so, like, haan, it was kind of difficult, but, I mean, we did jugaad, and somehow trained it, and yeah, I learned a lot, but still, like, next time we can do better, you know what I mean.'''
# recorded_phoneme = 'jɑːsoːʌmwʌntɛknikəltʃɛləndʒvʌzlaɪkudjlæŋɡwdʒjnoɪthædlaɪkvlɛsdɪtɜðɛnaðɛrssoːmɔdəlvʌzlaɪkkaɪndʌvstrʌɡlɪŋtʊlænɪtprɔpəliæmlaɪkjunoːðəlɔswʌzændɔlɔzwʌzhaɪənɔlændʌmɪthdʒʌstwʌzənvʌkŋɡudsoːmvtɹaɪttuːlaɪkfɪksttbaɪmeɪkɪŋbɛtʃɪsbɛlɛnstlaɪkjunomeɪkʃuːiːtʃbæhædtʊɾatʊɾfrʌmɛvridlɛŋɡwɪdʒsoːlaɪkiːvənɪfoːdiakadatakʌmtaɪtstɪlkeɪmɪntɹeɪnɪŋmɪnɪthɛldkaɪndʌvbʌtnɔtfulikɔzditkʌmtajuːnoːwʌtaɪminndsoːʌmjsoːtreɪnɪŋðəmɔdəlɪtsɛlfvʌzlaɪktʌfɪtstslaɪkarfɔstaɪmtɹeɪnɪŋsʌmtɪŋðɪsbɪmɛmərikakɑfiɪʃuːðiraðhænddʒipjbilimitədtasoːʌmvidɪdlaɪkðəɡreːdjəndɛkɪlmeɪʃənɛnstʌfændprsɛstdetaɪntʃaŋklaɪkʌsmɔlpɑːtsoːɪtdʌzəntkraɪsɔrsʌmtɪŋsoːlaɪkhaɪtwʌskaɪndʌvdɪfɪkəltbʌtaɪmiːnvidɪddʒɡɑːdændsʌmhaʊtendɪændjaaɪlndəlɔtbʌtstɪllaɪknɛkstaɪmvikəndʊbɛtərjunowʌtaɪmin'
# genAI/misspronounciation/dataset/achyut_pizza_false.wav
# 
# reference_text = 'I love eating pizza while watching documentaries about rural India'

import pandas as pd

phoneme_data = pd.read_csv('dataset/dataset.csv')


# recorded_phoneme = 'aɪloʊtiŋpiːzɐwaɪlwɔtɪŋdɔkkjʊməntɹizɐbaʊtruːrəlɪndiɐ'
# Took 18.5474112033844 seconds to retrieve logits. 0.053915880175101044 chars per sec

# reference_text = 'The engineer submited the data report on wednesday'
# recorded_phoneme = 'ðəɛndʒɪnjɪrsʌbmɪtədðədeɪdaripɔtɔnvɛrnɛsdi'
# Took 5.673489332199097 seconds to retrieve logits. 0.17625837318925403 chars per sec



lexicon, ref_words = generate_reference_phoneme(reference_text)
reference_phoneme =' '.join([phon for w, phon in lexicon])
# print(reference_phoneme)
# print(recorded_phoneme)


from sequence_align.pairwise import needleman_wunsch
seq_a = reference_phoneme
seq_b = list(recorded_phoneme.replace(' ',''))

# recorded_phoneme['text']
aligned_seq_a, aligned_seq_b = needleman_wunsch(
    seq_a,
    seq_b,
    match_score=1.0,
    mismatch_score=-1.0,
    indel_score=-1.0,
    gap="_",
)
aligned_reference_seq = ''.join(aligned_seq_a)
aligned_recorded_seq = ''.join(aligned_seq_b)

print("Data Point Number: ",sr)
print('Reference Text: ', reference_text)
print()
print('Reference Phoneme:',aligned_reference_seq)
print()
print('Recorded Phoneme: ', aligned_recorded_seq)
print()
print("Data Point",case)
print()

def find_word_start_positions(reference_sequence):
    # Split the sequence into words based on spaces
    words = reference_sequence.split()
    # Initialize a list to store the start positions
    start_positions = []
    # Initialize the current position
    current_position = 0
    # Iterate over the words
    for word in words:
        # Add the current position to the start positions list
        start_positions.append(current_position)
        # Increment the current position by the length of the word plus 1 (for the space)
        current_position += len(word) + 1
    return start_positions

def split_recorded_sequence(recorded_sequence, start_positions):
    # Initialize a list to store the split words
    split_words = []
    # Iterate over the start positions
    for i in range(len(start_positions)):
        # Get the start position
        start = start_positions[i]
        # If it's the last word, get the end position as the length of the sequence
        if i == len(start_positions) - 1:
            end = len(recorded_sequence)
        # Otherwise, get the end position as the start position of the next word
        else:
            end = start_positions[i + 1]
        # Extract the word from the recorded sequence
        word = recorded_sequence[start:end]
        # Add the word to the list
        split_words.append(word)
    return split_words
    
# recorded_sequence = "aɪ_hoːp_ðeɪ_hɛv_maɪ_fiːv__rədbrænd_aɪl_biː_bæk_su_n__tʊ_pliːz_w_iːdfoː__miː_"
ref_start_positions = find_word_start_positions(''.join(aligned_reference_seq))

# split recorded based on the reference start positions
rec_split_words = split_recorded_sequence(''.join(aligned_recorded_seq), ref_start_positions)
rec_split_words = [re.sub('( |\\_)$','',w) for w in rec_split_words]

# split ref based on the reference start positions
ref_split_words = split_recorded_sequence(''.join(aligned_reference_seq), ref_start_positions)
ref_split_words = [re.sub('(\\_| )$','',w) for w in ref_split_words]

# print('Reference Text: ',reference_text)
# print('(word, reference_phoneme, recorded_phoneme)',list(zip(ref_words, ref_split_words, rec_split_words)))
word_comparision_list = list(zip(ref_words, ref_split_words, rec_split_words))
# print(word_comparision_list)



table_data = []

for w, ref_w, rec_w in word_comparision_list:
    # Bold word
    word = f"\033[1m{w}\033[0m"
    
    # Color recognized phoneme
    if ref_w == rec_w:
        rec_string = f"\033[92m{rec_w}\033[0m"  # Green if matched
    else:
        mismatch_index = 0
        for i, (c1, c2) in enumerate(zip(ref_w, rec_w)):
            if c1 != c2:
                mismatch_index = i
                break
        rec_string = (
            f"{rec_w[:mismatch_index]}"
            f"\033[91m{rec_w[mismatch_index]}\033[0m"
            f"{rec_w[mismatch_index+1:]}"
        )
    
    # Match percentage
    matching = round(Levenshtein.ratio(ref_w, rec_w) * 100, 2)
    
    table_data.append([word, ref_w, rec_string, f"{matching:.2f}"])

# Print table using tabulate
headers = ["Word", "Reference", "Recognized", "Match %"]
print(tabulate(table_data, headers=headers, tablefmt="plain"))

from dotenv import load_dotenv
from openai import OpenAI
import os
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 

def call_llm(prompt: str, system:str = None,model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
    try:
        messages = []
        if system:
            messages = [{"role":"system","content":system}]
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error in call_llm func: {e}"
    
# system_message = """You are an expert dialect/accent coach for indian spoken english. you will provide valuable feedback to improve my indian accent. 
# For ease of understanding, I would prefer you give suggestions for mipronunciation using google pronunciation respelling.
# provide following Overall Impression, Specific Feedback, Google Pronunciation Respelling Suggestions, additional tips"""

system_message = 'Your task it to check if I am making any big pronounciation mistakes, please ignore small mistakes because it could be program error'
prompt = f"""Reference Text:  {reference_text} reference_phoneme {reference_phoneme}"""

# print(call_llm(prompt=prompt,system=system_message))