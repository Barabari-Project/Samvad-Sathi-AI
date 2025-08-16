from fastapi import FastAPI,UploadFile, File, HTTPException, Body, Request
from fastapi.responses import JSONResponse
from typing import Optional,Literal
from prompts.prompts import extract_resume_template,analyze_text_template,extract_knowledge_set_template,analyse_domain_template,Final_Summary_template
from prompts.gen_que_prompt import get_gen_que_prompt
from pydantic import BaseModel
import PyPDF2
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from g2p_en import G2p
import json
import time
import io
import os
from sarvamai import SarvamAI
from pacing import provide_pace_feedback
from pauses import analyze_pauses
import numpy as np
import librosa
import statistics
from typing import List, Dict
import replicate
import httpx
import re

# Create an HTTPX client with longer timeouts
custom_client = httpx.Client(timeout=httpx.Timeout(60.0))  # 5 min read timeout

from dotenv import load_dotenv
load_dotenv()

Sarvam_client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))
dg_client = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
g2p = G2p()

def call_llm(prompt: str, system:str = None,model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
    """
    Calls the specified language model (OpenAI or Sarvam) with a prompt and optional system message.

    Args:
        prompt (str): The user prompt to send to the model.
        system (str, optional): System instructions for the model.
        model (str): Model name ("gpt-4o-mini", "gpt-4o", or "sarvam").
        temperature (float): Sampling temperature for model output.

    Returns:
        str: The model's response as a string.
    """
    messages = []
    if system:
        messages = [{"role":"system","content":system}]
    messages.append({"role": "user", "content": prompt})
    if model == "gpt-4o-mini" or model == "gpt-4o":
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error in call_llm func: {e}"
    elif model == "sarvam":
        response = Sarvam_client.chat.completions(messages=messages,max_tokens=10_000)
        response = response.choices[0].message.content
        return response
    
class TextRequest(BaseModel):
    text: str
    
class dictRequest(BaseModel):
    dict_ : dict
    
class intRequest(BaseModel):
    num : int

    
    
def extract_json_dict(text: str):
    """
    Extracts and parses the first valid JSON object or array from a string.

    Args:
        text (str): Text containing JSON.

    Returns:
        dict or list: Parsed JSON object or array.

    Raises:
        ValueError: If no valid JSON is found.
    """
    try:
        start = min(
            (text.index('{') if '{' in text else float('inf')),
            (text.index('[') if '[' in text else float('inf'))
        )
        end = max(
            (text.rindex('}') + 1 if '}' in text else -1),
            (text.rindex(']') + 1 if ']' in text else -1)
        )
        json_str = text[start:end]

        # Remove trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

        # Escape raw newlines inside quoted strings
        def escape_newlines_in_strings(match):
            return match.group(0).replace("\n", "\\n")

        json_str = re.sub(r'"([^"\\]*(\\.[^"\\]*)*)"', escape_newlines_in_strings, json_str)
        
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print("LLM's OUTPUT")
        print(text)
        print("Extracted JSON string")
        print(json_str)
        raise ValueError(f"Invalid JSON found: {e}")


app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()  
    response = await call_next(request)  
    process_time = time.time() - start_time  
    response.headers["X-Process-Time"] = str(process_time) 
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class gen_que_model(BaseModel):
    extracted_resume : dict
    job_role : Literal["Data Science","Frontend Developer","Backend Developer"]
    number_of_ques : int
    job_description : Optional[str]
    years_of_exp : int

async def extract_resume_data(file: UploadFile) -> dict:
    """
    Extracts structured resume data from a PDF file using LLM.

    Args:
        file (UploadFile): PDF resume file.

    Returns:
        dict: Extracted resume data.

    Raises:
        HTTPException: If file is not PDF or extraction fails.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    try:
        contents = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
        text = "".join(page.extract_text() or "" for page in pdf_reader.pages).strip()
        json_str = call_llm(system=extract_resume_template, prompt=text)
        return extract_json_dict(json_str)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


def generate_questions(payload: gen_que_model) -> dict:
    """
    Generates interview questions using candidate's resume and job details.

    Args:
        payload (gen_que_model): Candidate profile and job info.
            extracted_resume : dict
            job_role : Literal["Data Science","Frontend Developer","Backend Developer"]
            number_of_ques : int
            job_description : Optional[str]
            years_of_exp : 
            
    Returns:
        dict: Generated interview questions in structured format.
    """
    prompt = get_gen_que_prompt(
        resume=payload.extracted_resume,
        YOE=payload.years_of_exp,
        JD=payload.job_description,
        Role=payload.job_role,
        NOQ=payload.number_of_ques,
    )

    system = "You generate structured interview questions based on a candidate's profile and job role. Output must follow the given JSON format."
    result = call_llm(system=system, prompt=prompt)
    return extract_json_dict(result)

@app.post("/extract-and-generate")
async def extract_and_generate_testing_api(
    file: UploadFile = File(...),
    job_role: Literal["Data Science", "Frontend Developer", "Backend Developer"] = ...,
    number_of_ques: int = 5,
    job_description: Optional[str] = None,
    years_of_exp: int = 0
):
    """
    Extracts resume data from a PDF and generates interview questions.

    Args:
        file (UploadFile): PDF resume file.
        job_role (str): Job role for interview questions.
        number_of_ques (int): Number of questions to generate.
        job_description (str, optional): Job description.
        years_of_exp (int): Years of experience.

    Returns:
        JSONResponse: Extracted resume and generated interview questions.
    """
    # Step 1: extract resume
    extracted_resume = await extract_resume_data(file)

    # Step 2: build input model for gen_questions
    payload = gen_que_model(
        extracted_resume=extracted_resume,
        job_role=job_role,
        number_of_ques=number_of_ques,
        job_description=job_description,
        years_of_exp=years_of_exp
    )

    # Step 3: generate questions
    questions = generate_questions(payload)

    return JSONResponse(content={
        "extracted_resume": extracted_resume,
        "interview_questions": questions
    })


@app.post("/extract-resume")
async def extract_text_from_pdf(file: UploadFile = File(...)):
    """
    Extracts structured resume data from a PDF file.

    Args:
        file (UploadFile): PDF resume file.

    Returns:
        JSONResponse: Extracted resume data.
    """
    extracted_resume = await extract_resume_data(file)
    return JSONResponse(content=extracted_resume)

@app.post('/generate-questions')
async def gen_questions(payload:gen_que_model = Body(...)):
    """
    Generates interview questions based on resume and job role.

    Args:
        payload (gen_que_model): Resume and job details.

    Returns:
        JSONResponse: Generated interview questions.
    """
    questions = generate_questions(payload)
    return JSONResponse(content={"interview_questions": questions})
    

@app.post("/transcribe_nova_3")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribes audio using Deepgram Nova-3 model. (Not optimal For our use case)

    Args:
        file (UploadFile): MP3 audio file.

    Returns:
        JSONResponse: Transcript, processing time, and chars per second.
    """
    if not file.filename.lower().endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Only .mp3 files are supported.")

    audio_bytes = await file.read()
    payload: FileSource = {"buffer": audio_bytes}

    options = PrerecordedOptions(
        model="nova-3",       
        smart_format=False,
        filler_words=True,
        language="multi",
        punctuate=False,
        utterances=True,
    )

    try:
        t1 = time.time()
        response = dg_client.listen.rest.v("1").transcribe_file(payload, options)
        t1 = time.time() - t1
        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        return JSONResponse(content={"transcript": transcript,"time":t1,"chars per sec":len(transcript)/t1})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    
def estimate_fluency_proxy(audio_bytes, transcript):
    """
    Estimates speech fluency using audio and transcript.

    Args:
        audio_bytes (bytes): Raw audio data.
        transcript (str): Transcript of the audio.

    Returns:
        tuple: (Words per minute (float), is_fluent (bool))
    """
    # Load audio from bytes
    audio_buffer = io.BytesIO(audio_bytes)
    audio_buffer.seek(0)
    y, sr = librosa.load(audio_buffer, sr=None)
    
    # Get voiced intervals
    intervals = librosa.effects.split(y, top_db=30)
    voiced_time_sec = sum([(end - start) / sr for start, end in intervals])
    
    word_count = len(transcript.strip().split())
    
    # Avoid division by zero
    if voiced_time_sec == 0:
        return 0, False
    
    wpm = word_count / (voiced_time_sec / 60)
    is_fluent = 90 <= wpm <= 200
    return round(wpm, 2), is_fluent

@app.post("/transcribe_whisper")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribes audio using OpenAI Whisper model.

    Args:
        file (UploadFile): Supported audio file (.mp3, .wav, .m4a, .flac).

    Returns:
        JSONResponse: Detailed transcription output.
    """
    # Validate file extension
    if not file.filename.lower().endswith((".mp3", ".wav", ".m4a", ".flac")):
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    
    audio_bytes = await file.read()
    buffer = io.BytesIO(audio_bytes)
    buffer.name = file.filename

    try:
        transcription = client.audio.transcriptions.create(
        model="whisper-1", 
        file=buffer, 
        response_format="verbose_json",
        prompt="matlab, jaise ki, vagera-vagera, I'm like,you know what I mean, kind of, um, ah, huh, and so, so um, uh, and um, like um, so like, like it's, it's like, i mean, yeah, ok so, uh so, so uh, yeah so, you know, it's uh, uh and, and uh, like, kind",
        timestamp_granularities=["word"]
        # include=["logprobs"]
        )
        transcription = transcription.model_dump()
        # new_transcript = ''
        # excluded = ''
        # for item in transcription['logprobs']:
        #     item["prob"] = np.exp(item["logprob"])
        #     if item['prob'] > 0.6:
        #         new_transcript += item['token']
        #     else:
        #         excluded += item['token']
        
        # transcription['new_transcript'] = new_transcript
        # transcription['excluded'] = excluded
    
        # wpm, is_fluent = estimate_fluency_proxy(audio_bytes, new_transcript)
        # transcription['fluency'] = {
        #     "wpm": float(wpm),
        #     "is_fluent": bool(is_fluent),
        #     "note": "Fluent" if is_fluent else "Too slow or too fast"
        # }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")
    
    return JSONResponse(content=transcription)

@app.post('/transcribe_whisperx')
async def transcribe_whisperx(audio_file_link:str = Body(...)):
    '''
    Transcribes the given audio file with wordlevel time stamps and confidance score
    
    Args:
        audio_file_link (str) : downloadable link to audio file
        
    Returns:
        JSONResponse: Detailed transcription output.
        
    Sample inputs:
    1) my_answer.wav = https://drive.google.com/uc?export=download&id=125S0xMf-J86RTGqIRtOEtwWt_IvYirYW
    2) bond_pause.mp3 = https://drive.google.com/uc?export=download&id=18IND7KyV2Od4QvvxvIJI8X3s06mosvb2
    3) Telugu audio with fillers.m4a = https://drive.google.com/uc?export=download&id=1VQY_bm2I0CJkYkjsxPMDmVrDlZe23HPA
    '''

    # pre = 'https://drive.google.com/uc?export=download&id='
    # audio_file_link = pre + '125S0xMf-J86RTGqIRtOEtwWt_IvYirYW' # my_answer.wav
    # audio_file_link = pre + '18IND7KyV2Od4QvvxvIJI8X3s06mosvb2' # bond_pause.mp3
    
    output = replicate.run(
        "victor-upmeet/whisperx:77505c700514deed62ab3891c0011e307f905ee527458afc15de7d9e2a3034e8",
        input={
            "debug": False,
            "vad_onset": 0.5,
            "audio_file": audio_file_link,
            "batch_size": 64,
            "vad_offset": 0.363,
            "diarization": False,
            "temperature": 0,
            "align_output": True
        }
    )

    words = []
    try:
        for i in output['segments']:
            words += i['words']
    except Exception as e:
        print("ERROR IN /transcribe_whisperx endpoint ",e)
        print(words)
        return JSONResponse(content={"words":{},'text':f"words = {words}"})
        
    transcript = ''
    for i in words:
        transcript += i['word'] + ' '
    transcript = transcript.strip()
    
    return JSONResponse(content={"words":words,'text':transcript})
        
    
@app.post('/get_knowledgeset')
async def get_knowledgeset(user_profile:dictRequest = Body(...)):
    """
    Extracts knowledge set from user profile skills.

    Args:
        user_profile (dictRequest): User profile with skills.

    Returns:
        JSONResponse: Extracted knowledge set.
    """
    user_profile = user_profile.dict_
    skills = user_profile.pop('skills')
    skills = str(skills)
    prompt = extract_knowledge_set_template.format(skills=skills)
    res = call_llm(prompt=prompt)
    res = extract_json_dict(res)
    return JSONResponse(content=res)

@app.post("/domain-base-analysis")
async def analyse_answer(
    user_profile : dict = Body(...),
    answer : str = Body(...),
    years_of_experience : int = Body(...),
    Interview_Question : dict = Body(...),
    job_role: Literal["Data Science", "Frontend Developer", "Backend Developer"] = Body(...),
):
    """
    Analyzes candidate's answer for domain knowledge.

    Args:
        user_profile (dict): User profile.
        answer (str): Candidate's answer.
        years_of_experience (int): Years of experience.
        Interview_Question (dict): Interview question details.
        job_role (str): Job role.

    Returns:
        JSONResponse: Domain analysis feedback.
    """
    user_profile.pop('name')
    user_profile.pop('contact')
    user_profile = str(user_profile)
    category,difficulty,question,hint = Interview_Question.values()
    
    prompt = analyse_domain_template.format(job_title=job_role,
                                          Years_of_experience=years_of_experience,
                                          Users_Resume_Profile=user_profile,
                                          Candidate_Response=answer,
                                          category=category,
                                          difficulty=difficulty,
                                          question=question,
                                          hint=hint
                                          )
    response = call_llm(prompt=prompt)
    json_res = extract_json_dict(response)
    return JSONResponse(content=json_res)

@app.post('/communication-based-analysis')
async def analyse_communication_features(answer:str = Body(...)):
    '''
        Evaluate the user-provided text across four key dimensions:  
        1. Clarity 
        2. Vocabulary richness  
        3. Grammar & syntax  
        4. Structure & flow  
        
        Args:
            answer (str): Candidate's answer.

        Returns:
            dict: Communication feedback.
    """
    '''
    result = call_llm(system=analyze_text_template,prompt=answer)
    result = extract_json_dict(result)
    return {"feedback":result}

@app.post('/pace-analysis')
async def measure_pace_features(words_timestamp:dict = Body()):
    '''
    Description:
    gives pace analysis of individual answer
    
    input:
    output of transcribe whisper api
    
    output:
    {
        'feedback': str,   # verbal feedback
        'score': float     # 0-100 pacing score
    }
    '''
    # provide_pace_feedback now returns a dictionary with keys
    # "feedback" (string) and "score" (float). We propagate both keys
    # to the client for richer information.
    res = provide_pace_feedback(words_timestamp)
    return JSONResponse(content=res)

@app.post('/pauses-analysis')
async def measure_pause_analysis(words_timestamp:dict = Body(...)): 
    """
    Analyzes pauses in spoken answer using word timestamps.

    Args:
        words_timestamp (dict): Word-level timestamp data.

    Returns:
        JSONResponse: Pause feedback.
    """   
    res = analyze_pauses(words_timestamp,call_llm=call_llm,extract_json_dict=extract_json_dict)
    return JSONResponse(content={"feedback":res})

import asyncio

@app.post("/complete-analysis")
async def complete_analysis(
    user_profile: dict = Body(...),
    answer: str = Body(...),
    years_of_experience: int = Body(...),
    Interview_Question: dict = Body(...),
    words_timestamp: dict = Body(...),
    job_role: Literal["Data Science", "Frontend Developer", "Backend Developer"] = Body(...),
):
    """
    Performs complete analysis of candidate's answer including domain, communication, pace, and pauses.

    Args:
        user_profile (dict): User profile.
        answer (str): Candidate's answer.
        years_of_experience (int): Years of experience.
        Interview_Question (dict): Interview question details.
        words_timestamp (dict): Word-level timestamp data.
        job_role (str): Job role.

    Returns:
        JSONResponse: Aggregated analysis results.
    """
    user_profile.pop('name', None)
    user_profile.pop('contact', None)
    user_profile_str = str(user_profile)
    category, difficulty, question, hint = Interview_Question.values()

    # Prepare prompt for domain analysis
    domain_prompt = analyse_domain_template.format(
        job_title=job_role,
        Years_of_experience=years_of_experience,
        Users_Resume_Profile=user_profile_str,
        Candidate_Response=answer,
        category=category,
        difficulty=difficulty,
        question=question,
        hint=hint
    )

    async def domain_analysis():
        response = await asyncio.to_thread(call_llm, prompt=domain_prompt)
        return extract_json_dict(response)

    async def communication_analysis():
        response = await asyncio.to_thread(call_llm, system=analyze_text_template, prompt=answer)
        return extract_json_dict(response)

    async def pace_analysis():
        return provide_pace_feedback(words_timestamp)

    async def pause_analysis():
        return analyze_pauses(words_timestamp, call_llm=call_llm,extract_json_dict=extract_json_dict)

    # Run all tasks concurrently
    domain_task, comm_task, pace_task, pause_task = await asyncio.gather(
        domain_analysis(),
        communication_analysis(),
        pace_analysis(),
        pause_analysis()
    )

    return JSONResponse(content={
        "Question":question,
        "Answer":answer,
        "domain_analysis": domain_task,
        "communication_analysis": comm_task,
        "pace_analysis": pace_task,
        "pause_analysis": pause_task
    })

@app.post('/final-report')
def genarate_final_report(Session_analysis:dict = Body(...)):
    '''
    Description:
    genarates final report at session level
    
    Input:
    {
        "analysis":[ele1,ele2...]
    }
    here each ele is output of complete analysis api for each questions in session
    
    Output:
    {
        "Summery":{},
        "knowledge_competence":{},
        "Speech_Structure_Fluency":{}
    }
    '''
    analysis_list = Session_analysis['analysis']
    if not analysis_list:
        return {"error": "No analysis data provided"}
    questions_list = [analysis["Question"] for analysis in analysis_list]
    answers_list = [analysis["Answer"] for analysis in analysis_list]
    pause_list = [analysis['pause_analysis'] for analysis in analysis_list]
    pace_list = [analysis['pace_analysis'] for analysis in analysis_list]
    communication_list = [analysis['communication_analysis'] for analysis in analysis_list]
    domian_list = [analysis['domain_analysis'] for analysis in analysis_list]
    
    Speech_Structure_Fluency = [{"pace":i,"pause":j,"communication":k} for i,j,k in zip(pace_list,pause_list,communication_list)]
    
    report = {"Summery":None,
            "knowledge_competence":domian_list,
              "Speech_Structure_Fluency": Speech_Structure_Fluency
              }

    acc_sum,dou_sum,rel_sum,example_sum = 0,0,0,0
    for domain in domian_list:
        acc_sum += domain["attribute_scores"]["Accuracy"]["score"]
        dou_sum += domain["attribute_scores"]["Depth of Understanding"]["score"]
        example_sum += domain["attribute_scores"]["Examples/Evidence"]["score"]
        rel_sum += domain["attribute_scores"]["Relevance"]["score"]

    
    clarity_sum,vocabulary_sum,grammar_sum,structure_sum = 0,0,0,0
    pace_sum, pause_sum = 0,0
    for ijk in Speech_Structure_Fluency:
        pace_item = ijk['pace']
        pause_item = ijk['pause']
        comm = ijk['communication']
        
        clarity_sum += comm["clarity"]["score"]
        vocabulary_sum += comm["vocabulary_richness"]["score"]
        grammar_sum += comm["grammar_syntax"]["score"]
        structure_sum += comm["structure_flow"]["score"]
        
        pace_sum += float(pace_item['score'])
        pause_sum += float(pause_item['score'])
        
    scores = {
        "knowledge_competence": {
            "Accuracy":acc_sum,
            "Depth of Understanding":dou_sum,
            "Examples/Evidence":example_sum,
            "Relevance":rel_sum
            },
        "Speech_Structure_Fluency":{
            "clarity":clarity_sum,
            "vocabulary_richness":vocabulary_sum,
            "grammar_syntax":grammar_sum,
            "structure_flow":structure_sum,
            "pace":pace_sum,
            "pause":pause_sum
        }
    }

    # Initialize enhanced aggregation structures
    knowledge_attributes = {
        'Accuracy': {'scores': [], 'reasons': []},
        'Depth of Understanding': {'scores': [], 'reasons': []},
        'Relevance': {'scores': [], 'reasons': []},
        'Examples/Evidence': {'scores': [], 'reasons': []}
    }
    
    communication_attributes = {
        'clarity': {'scores': [], 'rationales': [], 'quotes': []},
        'vocabulary_richness': {'scores': [], 'rationales': [], 'quotes': []},
        'grammar_syntax': {'scores': [], 'rationales': [], 'quotes': []},
        'structure_flow': {'scores': [], 'rationales': [], 'quotes': []}
    }
    
    wpm_values = []
    rushed_pause_percentages = []
    all_knowledge_feedbacks = []

    # Process each analysis entry
    for entry in analysis_list:
        # Domain analysis aggregation with reasons
        domain = entry['domain_analysis']
        for attr, info in domain['attribute_scores'].items():
            knowledge_attributes[attr]['scores'].append(info['score'])
            knowledge_attributes[attr]['reasons'].append(info['reason'])
        all_knowledge_feedbacks.append(domain['overall_feedback'])
        
        # Communication analysis with quotes
        comm = entry['communication_analysis']
        for category in communication_attributes.keys():
            cat_data = comm[category]
            communication_attributes[category]['scores'].append(cat_data['score'])
            communication_attributes[category]['rationales'].append(cat_data['rationale'])
            communication_attributes[category]['quotes'].extend(cat_data['quotes'])
        
        # Pace analysis extraction
        pace_str = entry['pace_analysis']
        if "Your average pace:" in pace_str:
            wpm_line = next(line for line in pace_str.split('\n') if "Your average pace:" in line)
            wpm_values.append(float(wpm_line.split(":")[2].split()[0]))
        
        # Pause analysis extraction
        pause = entry['pause_analysis']
        rushed_pct = float(pause['distribution']['rushed'].strip('%'))
        rushed_pause_percentages.append(rushed_pct)

    # Calculate averages
    avg_knowledge_scores = {attr: statistics.mean(data['scores']) for attr, data in knowledge_attributes.items()}
    avg_comm_scores = {attr: statistics.mean(data['scores']) for attr, data in communication_attributes.items()}
    avg_wpm = statistics.mean(wpm_values) if wpm_values else 0
    avg_rushed_pause = statistics.mean(rushed_pause_percentages) if rushed_pause_percentages else 0

    # Prepare enhanced LLM prompt with context
    knowledge_section = "=== KNOWLEDGE PERFORMANCE ===\nAverage Scores (1-5 scale):\n"
    for attr, data in knowledge_attributes.items():
        avg_score = statistics.mean(data['scores'])
        unique_reasons = set(data['reasons'])  # Deduplicate while preserving context
        knowledge_section += f"- {attr}: {avg_score:.1f}\n"
        knowledge_section += f"  Reasons:\n"
        for reason in unique_reasons:
            knowledge_section += f"  • {reason}\n"
        knowledge_section += "\n"
    
    knowledge_section += "Key Feedback Themes:\n"
    for fb in set(all_knowledge_feedbacks):
        knowledge_section += f"- {fb}\n"
    
    comm_section = "\n\n=== COMMUNICATION PERFORMANCE ===\nAverage Scores (1-5 scale):\n"
    for category, data in communication_attributes.items():
        avg_score = statistics.mean(data['scores'])
        comm_section += f"- {category.replace('_', ' ').title()}: {avg_score:.1f}\n"
        
        # Add representative quotes
        unique_quotes = set(data['quotes'])
        if unique_quotes:
            comm_section += f"  Example Quotes:\n"
            for quote in list(unique_quotes)[:3]:  # Limit to 3 most representative
                comm_section += f"  • \"{quote}\"\n"
        
        # Add unique rationales
        unique_rationales = set(data['rationales'])
        comm_section += f"  Rationales:\n"
        for rationale in unique_rationales:
            comm_section += f"  • {rationale}\n"
        comm_section += "\n"
    
    prompt = Final_Summary_template.format(
        knowledge_section=knowledge_section,
        comm_section=comm_section,
        avg_wpm=avg_wpm,
        avg_rushed_pause=avg_rushed_pause,
    )

    Final_Summary = call_llm(prompt,model="gpt-4o")
    
    report['Summery'] = {
        "Scores":scores,
        "Final Summary":Final_Summary
    }
    
    return JSONResponse(content=report)