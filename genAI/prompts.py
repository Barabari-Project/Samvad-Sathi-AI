analyze_text_template='''
**System Role**  
Act as a linguistic analysis expert. Evaluate the user-provided text across four key dimensions:  
1. **Clarity**  
2. **Vocabulary richness**  
3. **Grammar & syntax**  
4. **Structure & flow**  

**Output Requirements**  
Return JSON format with these keys:  
- `clarity`  
- `vocabulary_richness`  
- `grammar_syntax`  
- `structure_flow`  

For each key, provide:  
- `score` (1-5 scale: 1=Poor, 5=Excellent)  
- `rationale` (2-3 sentence explanation with evidence, also )  
- `quotes` (1-3 representative text excerpts illustrating the assessment)  

**Analysis Guidelines**  
| Feature          | Evaluation Criteria                                                                 |  
|------------------|-------------------------------------------------------------------------------------|  
| Clarity          | Sentence coherence, conciseness, ambiguity avoidance, filler usage                  |  
| Vocabulary       | Lexical diversity, context-appropriateness, sophistication, repetition analysis     |  
| Grammar & Syntax | Grammatical correctness, tense consistency, punctuation, sentence structure fluency |  
| Structure & Flow | Logical progression, paragraph transitions, thematic cohesion, argument sequencing |  
'''

extract_resume_template = f'''
your task is to extract json with keys
  "name",
  "contact",
  "education",
  "experience",
  "certifications",
  "projects",
  "skills"

from resume text. if information in resume is incomplete based on the keys, keep the values of key empty.
'''

gen_question_template='''
Role: Expert interviewer for {target_role}.

Context:
- Candidate Profile: {relevent_info}
{job_highlights}

TASK: 
genarate n = {n} interview questions following the instructions.

Instructions:
IF n == 1:
  - Generate 1 question that you believe is the MOST insightful, based on the candidate's profile and the role.
  - The question can be technical, behavioral, or resume-specific.
ELSE (if n > 1):
  - Generate {n} questions covering the following distribution:
    - Technical skills (~40%)
    - Behavioral/situational scenarios (~30%)
    - Resume-specific deep dives (~20%)
    - Role-specific knowledge (~10%)
  - Phrase questions conversationally (e.g., "Tell me about a time...").
  - Include 1-2 challenge questions targeting experience gaps.

Output format: JSON list with "question" and "category" keys.
'''


analyze_answer_template='''
Role: Expert Interview Analyst
Context:

- Target Role: {job_title}
- Seniority Level: {level}
- User's resume: {user_profile}
 
- Question: "{interview_question}"

Candidate Response: "{user_response}"

Evaluation Tasks:

1. Rate each dimension 1-5 (5=excellent):
    - Relevance: _
    - Completeness: _
    - Clarity: _
    - Depth: _
    - Authenticity: _
    - Skill Demonstration: _
2. Identify top 2 strengths with specific examples
3. Identify top 2 improvement areas with actionable advice
4. Provide overall feedback (1-2 sentences)
5. Generate follow-up question (if needed)

Output Format (JSON):
{{
"scores": {{dimension: score}},
"strengths": ["strength1", "strength2"],
"improvements": ["advice1", "advice2"],
"overall_feedback": "text",
"follow_up_question": "text"
}}
'''