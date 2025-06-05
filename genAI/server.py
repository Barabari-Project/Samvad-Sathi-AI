from fastapi import FastAPI
from dotenv import load_dotenv
import os
from agents import Agent,Runner
from prompts import check_language
from pydantic import BaseModel
import inspect

load_dotenv()
api_key = os.getenv("api_key")


language_agent = Agent(
    name="language_agent",
    model="gpt-4o-mini",
    instructions=check_language,
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
    print(payload.text)
    inspect.signature(Agent)
    return {"feedback":result.final_output}