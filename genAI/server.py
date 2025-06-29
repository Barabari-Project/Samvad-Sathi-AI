from fastapi import FastAPI,UploadFile, File, HTTPException, Body, Request
from fastapi.responses import JSONResponse
from typing import Optional
from dotenv import load_dotenv
from prompts import extract_resume_template,gen_question_template,analyze_text_template,analyze_answer_template,extract_knowledge_set_template
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
from genAI.pacing import provide_pace_feedback
from genAI.pauses import analyze_pauses

load_dotenv()
Sarvam_client = SarvamAI(
    api_subscription_key=os.getenv("SARVAM_API_KEY"),
)

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
    profile : dict
    
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

@app.post("/extract-resume")
async def extract_text_from_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        contents = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        text = text.strip()
        json_str = call_llm(system=extract_resume_template,prompt=text)
        dic = extract_json_dict(json_str)
        return JSONResponse(content=dic)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
        
@app.post("/gen-questions")
async def gen_questions(user_profile:dictRequest,target_job:TextRequest,number_of_ques:intRequest,job_description:Optional[TextRequest] = Body(default=None)):
    n = str(number_of_ques.num)
    if job_description:
        job_description = "- Job Requirements: " + job_description.text
    else:
        job_description = ''
    prompt = gen_question_template.format(n=n,relevent_info=str(user_profile.profile),job_highlights=job_description,target_role=target_job.text)
    system="You generate structured interview questions based on a candidate's profile and job role. Output must follow the given JSON format."
    result = call_llm(system=system,prompt=prompt)
    ex_json = extract_json_dict(result)
    return JSONResponse(content=ex_json)

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

# 
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
        response_format="verbose_json", # verbose_json
        prompt="matlab, jaise ki, vagera-vagera, I'm like,you know what I mean, kind of, um, ah, huh, and so, so um, uh, and um, like um, so like, like it's, it's like, i mean, yeah, ok so, uh so, so uh, yeah so, you know, it's uh, uh and, and uh, like, kind",
        timestamp_granularities=["word"]
        )
        transcription = transcription.model_dump()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")
    
    return JSONResponse(content=transcription)
    

@app.post("/extract-resume-and-gen-questions")
async def extract_resume_and_gen_questions(
    file: UploadFile = File(...),
    target_job: str = Body(...),
    number_of_ques: int = Body(...),
    job_description: Optional[str] = Body(default=None)
):
    """
    Combined endpoint that extracts resume data from PDF and generates interview questions
    
    Args:
        file: PDF file containing the resume
        target_job: Target job role for the candidate
        number_of_ques: Number of questions to generate
        job_description: Optional job description/requirements
    
    Returns:
        JSON containing both extracted resume data and generated questions
    """
    
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        # Step 1: Extract resume data from PDF
        contents = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
        text = ""
        
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        
        text = text.strip()
        
        # Extract structured resume data using the agent
        json_str = call_llm(system=extract_resume_template,prompt=text)
        resume_data = extract_json_dict(json_str)
        
        # Step 2: Generate questions based on extracted resume data
        n = str(number_of_ques)
        
        if job_description:
            job_description_formatted = "- Job Requirements: " + job_description
        else:
            job_description_formatted = ''
        
        prompt = gen_question_template.format(
            n=n,
            relevent_info=str(resume_data),
            job_highlights=job_description_formatted,
            target_role=target_job
        )
        
        # Generate questions using the agent
        system="You generate structured interview questions based on a candidate's profile and job role. Output must follow the given JSON format."
        result = call_llm(system=system,prompt=prompt)
        questions_data = extract_json_dict(result)
        
        # Step 3: Return combined response
        response = {
            "resume_data": resume_data,
            "questions": questions_data,
            "metadata": {
                "target_job": target_job,
                "number_of_questions": number_of_ques,
                "has_job_description": job_description is not None
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process resume and generate questions: {str(e)}"
        )
        
@app.post('/communication-based-analysis')
async def analyse_text_response(payload: TextRequest):
    '''
        Evaluate the user-provided text across four key dimensions:  
        1. Clarity 
        2. Vocabulary richness  
        3. Grammar & syntax  
        4. Structure & flow  
    '''
    result = call_llm(system=analyze_text_template,prompt=payload.text)
    result = extract_json_dict(result)
    return {"feedback":result}


class AnalyseAnswerRequest(BaseModel):
    question : str
    answer : str
    target_job_role : str
    seniority_level : str
    
@app.post("/domain-base-analysis")
async def analyse_answer(user_profile:dictRequest = Body(...), payload: AnalyseAnswerRequest = Body(...)):
    user_profile = user_profile.profile
    user_profile.pop('name')
    user_profile.pop('contact')
    user_profile = str(user_profile)
    
    prompt = analyze_answer_template.format(job_title=payload.target_job_role,
                                          level=payload.seniority_level,
                                          user_profile=user_profile,
                                          interview_question=payload.question,
                                          user_response=payload.answer)
    response = call_llm(prompt=prompt)
    json_res = extract_json_dict(response)
    return JSONResponse(content=json_res)

@app.post('/get_knowledgeset')
async def get_knowledgeset(user_profile:dictRequest = Body(...)):
    user_profile = user_profile.profile
    skills = user_profile.pop('skills')
    skills = str(skills)
    prompt = extract_knowledge_set_template.format(skills=skills)
    res = call_llm(prompt=prompt)
    res = extract_json_dict(res)
    return JSONResponse(content=res)

@app.post('/misspronounciation-analysis')
async def measure_paralinguistic_features(words:dictRequest = Body()):
    words = words.profile

    res = provide_pace_feedback(words)
    
    return JSONResponse(content={"feedback":res})

@app.post('/pauses-analysis')
async def do_pauses_analysis(words:dictRequest):
    words = words.profile
    
    res = analyze_pauses(words,call_llm=call_llm)
    
    return JSONResponse(content={"feedback":res})
