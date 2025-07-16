from fastapi import FastAPI,UploadFile, File, HTTPException, Body, Request
from fastapi.responses import JSONResponse
from typing import Optional,Literal
from prompts.prompts import extract_resume_template,analyze_text_template,extract_knowledge_set_template,analyse_domain_template
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


from dotenv import load_dotenv
load_dotenv()

Sarvam_client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))
dg_client = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
g2p = G2p()

def call_llm(prompt: str, system:str = None,model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
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
    dict : dict
    
class intRequest(BaseModel):
    num : int

    
    
def extract_json_dict(text: str):
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
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(text)
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
    extracted_resume = await extract_resume_data(file)
    return JSONResponse(content=extracted_resume)

@app.post('/generate-questions')
async def gen_questions(payload:gen_que_model = Body(...)):
    questions = generate_questions(payload)
    return JSONResponse(content={"interview_questions": questions})
    

@app.post("/transcribe_nova_3")
async def transcribe_audio(file: UploadFile = File(...)):
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
    
@app.post('/get_knowledgeset')
async def get_knowledgeset(user_profile:dictRequest = Body(...)):
    user_profile = user_profile.dict
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
    '''
    result = call_llm(system=analyze_text_template,prompt=answer)
    result = extract_json_dict(result)
    return {"feedback":result}

@app.post('/pace-analysis')
async def measure_pace_features(words_timestamp:dict = Body()):
    res = provide_pace_feedback(words_timestamp)
    return JSONResponse(content={"feedback":res})

@app.post('/pauses-analysis')
async def measure_pause_analysis(words_timestamp:dict = Body(...)):    
    res = analyze_pauses(words_timestamp,call_llm=call_llm)
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
        return analyze_pauses(words_timestamp, call_llm=call_llm)

    # Run all tasks concurrently
    domain_task, comm_task, pace_task, pause_task = await asyncio.gather(
        domain_analysis(),
        communication_analysis(),
        pace_analysis(),
        pause_analysis()
    )

    return JSONResponse(content={
        "domain_analysis": domain_task,
        "communication_analysis": comm_task,
        "pace_analysis": pace_task,
        "pause_analysis": pause_task
    })

@app.post('/final-report')
def genarate_final_report(Session_analysis:dict = Body(...)):
    analysis_list = Session_analysis['analysis']
    pause_list = [analysis['pause_analysis'] for analysis in analysis_list]
    pace_list = [analysis['pace_analysis'] for analysis in analysis_list]
    communication_list = [analysis['communication_analysis'] for analysis in analysis_list]
    domian_list = [analysis['domain_analysis'] for analysis in analysis_list]
    analysis_list = None
    
    Speech_Structure_Fluency = [{"pace":i,"pause":j,"communication":k} for i,j,k in zip(pace_list,pause_list,communication_list)]
    
    report = {"Summery":None,
        "knowledge_competence":domian_list,
              "Speech_Structure_Fluency": Speech_Structure_Fluency}
    

    acc_sum,dou_sum,rel_sum,example_sum = 0,0,0,0
    for domain in domian_list:
        acc_sum += domain["attribute_scores"]["Accuracy"]["score"]
        dou_sum += domain["attribute_scores"]["Depth of Understanding"]["score"]
        example_sum += domain["attribute_scores"]["Examples/Evidence"]["score"]
        rel_sum += domain["attribute_scores"]["Relevance"]["score"]    
    
    clarity_sum,vocabulary_sum,grammar_sum,structure_sum = 0,0,0,0
    for comm in communication_list:
        clarity_sum += comm["clarity"]["score"]
        vocabulary_sum += comm["vocabulary_richness"]["score"]
        grammar_sum += comm["grammar_syntax"]["score"]
        structure_sum += comm["structure_flow"]["score"]
    
    scores = {"scores":{
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
            "structure_flow":structure_sum
        }
    }}
    
    
    
    report['Summery'] = {
        "Scores":scores,
    }
    
    return JSONResponse(content=report)
    
    