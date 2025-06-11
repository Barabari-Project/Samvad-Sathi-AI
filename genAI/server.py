from fastapi import FastAPI,UploadFile, File, HTTPException, Body, Request
from fastapi.responses import JSONResponse
from typing import Optional
from dotenv import load_dotenv
import os
from agents import Agent,Runner
from prompts import check_language,extract_resume,gen_question_template
from pydantic import BaseModel
import PyPDF2
import io
import json
import re
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
import time

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
dg_client = DeepgramClient(DEEPGRAM_API_KEY)

language_agent = Agent(
    name="language_agent",
    model="gpt-4o-mini",
    instructions=check_language,
)

extract_resume_agent = Agent(
    name="extract resume",
    model="gpt-4o-mini",
    instructions=extract_resume,
)

gen_questions_agent = Agent(
    name="questions genarator",
    model="gpt-4o-mini",
    instructions="You generate structured interview questions based on a candidate's profile and job role. Output must follow the given JSON format.",
)

class TextRequest(BaseModel):
    text: str
    
class dictRequest(BaseModel):
    dic : dict
    
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

@app.post('/check-language')
async def check_lang(payload: TextRequest):
    result = await Runner.run(language_agent,"**Text to Analyze** \n  {payload.text}")
    result = extract_json_dict(result.final_output)
    return {"feedback":result}

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
        json_str = await Runner.run(extract_resume_agent,text)
        json_str = json_str.final_output
        dic = extract_json_dict(json_str)
        # json_str = {k: str(v) for k, v in dic.items()}
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
    prompt = gen_question_template.format(n=n,relevent_info=str(user_profile.dic),job_highlights=job_description,target_role=target_job.text)

    result = await Runner.run(gen_questions_agent,prompt)
    result = result.final_output
    json = extract_json_dict(result)
    return JSONResponse(content=json)


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Only .mp3 files are supported.")

    audio_bytes = await file.read()
    payload: FileSource = {"buffer": audio_bytes}

    options = PrerecordedOptions(
        model="nova-3",       
        smart_format=False
    )

    try:
        t1 = time.time()
        response = dg_client.listen.rest.v("1").transcribe_file(payload, options)
        t1 = time.time() - t1
        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        return JSONResponse(content={"transcript": transcript,"time":t1,"chars per sec":len(transcript)/t1})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

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
        json_str = await Runner.run(extract_resume_agent, text)
        json_str = json_str.final_output
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
        result = await Runner.run(gen_questions_agent, prompt)
        result = result.final_output
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