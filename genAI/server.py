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
    dict_ : dict
    
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
        "Question":question,
        "Answer":answer,
        "domain_analysis": domain_task,
        "communication_analysis": comm_task,
        "pace_analysis": pace_task,
        "pause_analysis": pause_task
    })

@app.post('/final-report')
def genarate_final_report(Session_analysis:dict = Body(...)):
    analysis_list = Session_analysis['analysis']
    print(analysis_list)
    num_ques = len(analysis_list)
    questions_list = [analysis["Question"] for analysis in analysis_list]
    answers_list = [analysis["Answer"] for analysis in analysis_list]
    pause_list = [analysis['pause_analysis'] for analysis in analysis_list]
    pace_list = [analysis['pace_analysis'] for analysis in analysis_list]
    communication_list = [analysis['communication_analysis'] for analysis in analysis_list]
    domian_list = [analysis['domain_analysis'] for analysis in analysis_list]
    analysis_list = None
    
    Speech_Structure_Fluency = [{"pace":i,"pause":j,"communication":k} for i,j,k in zip(pace_list,pause_list,communication_list)]
    
    report = {"Summery":None,
            "knowledge_competence":domian_list,
              "Speech_Structure_Fluency": Speech_Structure_Fluency
              }
    
    domain_attribute_prompt_template = '''
    ###
    <Question-{num}>
    {question}
    
    <ANS-{num}>
    {answer}
    
    <{attribute} analysis>
    {analysis}
    '''

    acc_sum,dou_sum,rel_sum,example_sum = 0,0,0,0
    acc_rea,dou_rea,rel_rea,example_rea = "","","",""
    for ix,(question,answer,domain) in enumerate(zip(questions_list,answers_list,domian_list)):
        acc_sum += domain["attribute_scores"]["Accuracy"]["score"]
        dou_sum += domain["attribute_scores"]["Depth of Understanding"]["score"]
        example_sum += domain["attribute_scores"]["Examples/Evidence"]["score"]
        rel_sum += domain["attribute_scores"]["Relevance"]["score"]
        
        acc_rea += domain_attribute_prompt_template.format(
            num=str(ix),
            question=question,
            answer=answer,
            attribute="accuracy",
            analysis=domain["attribute_scores"]["Accuracy"]["reason"])
        
        dou_rea += domain_attribute_prompt_template.format(
            num=str(ix),
            question=question,
            answer=answer,
            attribute="depth of understanding",
            analysis=domain["attribute_scores"]["Depth of Understanding"]["reason"]
        )

        rel_rea += domain_attribute_prompt_template.format(
            num=str(ix),
            attribute="relevance",
            question=question,
            answer=answer,
            analysis=domain["attribute_scores"]["Relevance"]["reason"]
        )

        example_rea += domain_attribute_prompt_template.format(
            num=str(ix),
            attribute="examples/evidence",
            question=question,
            answer=answer,
            analysis=domain["attribute_scores"]["Examples/Evidence"]["reason"]
        )
        
    comm_attribute_prompt_template = '''
    ###
    <Question-{num}>
    
    <answer>
    {answer}
    
    <Qoutes>
    {Qoutes}
    
    <{attribute}-analysis>
    {analysis}
    '''
    
    clarity_sum,vocabulary_sum,grammar_sum,structure_sum = 0,0,0,0
    clarity_rea,vocab_rea,grammar_rea,struct_rea = "","","",""
    for comm in communication_list:
        clarity_sum += comm["clarity"]["score"]
        vocabulary_sum += comm["vocabulary_richness"]["score"]
        grammar_sum += comm["grammar_syntax"]["score"]
        structure_sum += comm["structure_flow"]["score"]
            
        clarity_rea += comm_attribute_prompt_template.format(
            num=str(ix),
            attribute="clarity",
            answer=answer,
            analysis=comm["clarity"]["rationale"],
            Qoutes=comm["clarity"]["quotes"]
        )

        vocab_rea += comm_attribute_prompt_template.format(
            num=str(ix),
            attribute="vocabulary richness",
            answer=answer,
            analysis=comm["vocabulary_richness"]["rationale"],
            Qoutes=comm["vocabulary_richness"]["quotes"]
        )

        grammar_rea += comm_attribute_prompt_template.format(
            num=str(ix),
            attribute="grammar and syntax",
            answer=answer,
            analysis=comm["grammar_syntax"]["rationale"],
            Qoutes=comm["grammar_syntax"]["quotes"]
        )

        struct_rea += comm_attribute_prompt_template.format(
            num=str(ix),
            attribute="structure and flow",
            answer=answer,
            analysis=comm["structure_flow"]["rationale"],
            Qoutes=comm["structure_flow"]["quotes"]
        )
            
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
    
    # Thresholds
    strength_threshold = 2.5

    # Initialize output lists
    strengths = []
    areas_for_improvement = []

    # Traverse each category and attribute
    for category, attributes in scores["scores"].items():
        for attr, total_score in attributes.items():
            avg_score = total_score / num_ques
            if avg_score >= strength_threshold:
                strengths.append(attr)
            else:
                areas_for_improvement.append(attr)

    # # Print or use results
    # print(scores)
    # print("Strengths:", strengths)
    # print("Areas for Improvement:", areas_for_improvement)
    
     
            
    
    report['Summery'] = {
        "Scores":scores,
    }
    
    return JSONResponse(content=report)
    
    
import statistics
from typing import List, Dict

@app.post('/final-report-2')
def generate_final_report(Session_analysis: dict = Body(...)):
    analysis_list = Session_analysis['analysis']
    if not analysis_list:
        return {"error": "No analysis data provided"}

    # Initialize aggregation structures
    knowledge_attributes = {
        'Accuracy': [],
        'Depth of Understanding': [],
        'Relevance': [],
        'Examples/Evidence': []
    }
    communication_attributes = {
        'clarity': [],
        'vocabulary_richness': [],
        'grammar_syntax': [],
        'structure_flow': []
    }
    wpm_values = []
    rushed_pause_percentages = []
    all_knowledge_feedbacks = []
    all_comm_feedbacks = []

    # Process each analysis entry
    for entry in analysis_list:
        # Domain analysis aggregation
        domain = entry['domain_analysis']
        for attr, info in domain['attribute_scores'].items():
            knowledge_attributes[attr].append(info['score'])
        all_knowledge_feedbacks.append(domain['overall_feedback'])
        
        # Communication analysis aggregation
        comm = entry['communication_analysis']
        for category in communication_attributes.keys():
            communication_attributes[category].append(comm[category]['score'])
            all_comm_feedbacks.append(comm[category]['rationale'])
        
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
    avg_knowledge_scores = {attr: statistics.mean(scores) for attr, scores in knowledge_attributes.items()}
    avg_comm_scores = {attr: statistics.mean(scores) for attr, scores in communication_attributes.items()}
    avg_wpm = statistics.mean(wpm_values) if wpm_values else 0
    avg_rushed_pause = statistics.mean(rushed_pause_percentages) if rushed_pause_percentages else 0

    # Prepare LLM prompt
    prompt = f"""
Generate a comprehensive interview performance report using the following aggregated metrics:

=== KNOWLEDGE PERFORMANCE ===
Average Scores (1-5 scale):
- Accuracy: {avg_knowledge_scores['Accuracy']:.1f}
- Depth of Understanding: {avg_knowledge_scores['Depth of Understanding']:.1f}
- Relevance: {avg_knowledge_scores['Relevance']:.1f}
- Examples/Evidence: {avg_knowledge_scores['Examples/Evidence']:.1f}

Key Feedback Themes:
{chr(10).join('- ' + fb for fb in all_knowledge_feedbacks)}

=== COMMUNICATION PERFORMANCE ===
Average Scores (1-5 scale):
- Clarity: {avg_comm_scores['clarity']:.1f}
- Vocabulary: {avg_comm_scores['vocabulary_richness']:.1f}
- Grammar: {avg_comm_scores['grammar_syntax']:.1f}
- Structure: {avg_comm_scores['structure_flow']:.1f}

Key Feedback Themes:
{chr(10).join('- ' + fb for fb in all_comm_feedbacks)}

=== SPEECH METRICS ===
- Average Words/Minute: {avg_wpm:.1f} (Target: 120-150 WPM)
- Rushed Transitions: {avg_rushed_pause:.1f}% of phrases

=== REPORT STRUCTURE ===
🧾 Final Summary (with Actionable Steps)

✅ Strengths
Knowledge-Related:
- Highlight demonstrated technical understanding
- Note relevant terminology usage
- Mention strongest knowledge attributes

Speech Fluency-Related:
- Identify effective communication patterns
- Note positive aspects of pacing/structure
- Highlight vocabulary strengths

❌ Areas for Improvement
Knowledge-Related:
- Point out conceptual gaps
- Identify areas needing deeper examples
- Note inconsistencies in explanations

Speech Fluency-Related:
- Highlight filler word usage
- Note grammar/syntax challenges
- Identify structural issues
- Address pacing concerns

🎯 Actionable Steps
For Knowledge Development:
- Create 2-3 specific study recommendations based on knowledge gaps
- Suggest practical exercise types

For Speech & Structure:
- Recommend 2-3 targeted fluency exercises
- Include specific grammar/structure drills
- Provide pacing improvement strategies
"""

    # Generate final report using LLM
    report = call_llm(prompt)
    print(prompt)
    return {"report": report}

import statistics
from typing import List, Dict

@app.post('/final-report-v4')
def generate_final_report(Session_analysis: dict = Body(...)):
    analysis_list = Session_analysis['analysis']
    if not analysis_list:
        return {"error": "No analysis data provided"}

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
    
    prompt = f"""
Generate a comprehensive interview performance report using the following detailed metrics:

{knowledge_section}

{comm_section}

=== SPEECH METRICS ===
- Average Words/Minute: {avg_wpm:.1f} (Target: 120-150 WPM)
- Rushed Transitions: {avg_rushed_pause:.1f}% of phrases

=== REPORT STRUCTURE ===
🧾 Final Summary (with Actionable Steps)

✅ Strengths
Knowledge-Related:
- Highlight demonstrated technical understanding with specific examples from quotes
- Note relevant terminology usage from example quotes
- Mention strongest knowledge attributes based on reason analysis

Speech Fluency-Related:
- Identify effective communication patterns from positive examples
- Note positive aspects of pacing/structure from metrics
- Highlight vocabulary strengths from example quotes

❌ Areas for Improvement
Knowledge-Related:
- Point out conceptual gaps using specific reasons
- Identify areas needing deeper examples using feedback context
- Note inconsistencies in explanations using example quotes

Speech Fluency-Related:
- Highlight filler word usage with specific quotes
- Note grammar/syntax challenges with example errors
- Identify structural issues using rationales
- Address pacing concerns using WPM metrics

🎯 Actionable Steps
For Knowledge Development:
- Create 2-3 specific study recommendations based on knowledge gaps and reasons
- Suggest practical exercise types using feedback context

For Speech & Structure:
- Recommend 2-3 targeted fluency exercises using specific quotes
- Include specific grammar/structure drills based on error examples
- Provide pacing improvement strategies using WPM analysis
"""

    # Generate final report using LLM
    report = call_llm(prompt,model="gpt-4o")
    print(prompt)
    return {"report": report}