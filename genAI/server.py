from fastapi import FastAPI,UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
from agents import Agent,Runner
from prompts import check_language,extract_resume
from pydantic import BaseModel
import PyPDF2
import io
import json
import re

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

class TextRequest(BaseModel):
    text: str

@app.post('/check-language')
async def check_lang(payload: TextRequest):
    result = await Runner.run(language_agent,"**Text to Analyze** \n  {payload.text}")
    return {"feedback":result.final_output}

def extract_json_dict(text: str) -> dict:
    try:
        start = text.index('{')
        end = text.rindex('}') + 1
        json_str = text[start:end]
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON found: {e}")

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
        return JSONResponse(content={"json-resume":dic})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    

