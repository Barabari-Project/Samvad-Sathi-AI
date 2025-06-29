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
- `rationale` (2-3 sentence explanation with evidence, also make sure it is Easy-to-understand English)  
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
make sure to extract all projects from resume irrespective to the heading under which its mentioned
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

# technincal depth and techninal relevenace , Seniority Appropriateness
# elaborate on Seniority
# YOE instead of level for Seniority Appropriateness
# response should change according to the catagory of question
# analyze_answer_template for overall interview
analyze_answer_template = '''
Role: Expert Interview Analyst
Context:
- Target Role: {job_title} (Seniority: {level})
- User's Resume Profile: "{user_profile}"

- Interview Question: "{interview_question}"
- Candidate Response: "{user_response}"

Evaluation Tasks:
1. Rate dimensions Score:(1-5) (5=excellent) **relative to profile and role expectations**:
   - **Profile Alignment**: How well response maps to resume skills/experience (5=direct evidence)
   - **Role Relevance**: Fit for {job_title} responsibilities (5=perfect match)
   - **Seniority Appropriateness**: Depth expected for {level} level (5=exceeds level)
   - **Evidence Quality**: Specificity of examples from profile (5=quantifiable proof)
   - **Growth Demonstration**: Shows progression beyond resume (5=clear evolution)

2. Strengths (Top 2): 
   - Focus on **profile-specific advantages** (e.g., "Leveraged [resume skill] effectively in...")
   - Highlight **role-critical strengths** (e.g., "Demonstrated {job_title}-critical skill in...")

3. Improvements (Top 2):
   - **Profile-grounded advice** (e.g., "Expand on [resume bullet point] with metrics...")
   - **Role-specific gaps** (e.g., "For {level} role, add strategic perspective on...")

4. Overall Feedback: Directly address **profile-to-role fit** (1-2 sentences)

5. Follow-up Question: Probe **profile/role contradictions** or **resume opportunities**

Output Format (JSON):
{{
  "scores": {{
    "Profile Alignment": _,
    "Role Relevance": _,
    "Seniority Appropriateness": _,
    "Evidence Quality": _,
    "Growth Demonstration": _
  }},
  "strengths": ["[Profile-specific strength] + resume evidence", "[Role-critical strength]"],
  "improvements": ["[Profile-specific advice] + resume reference", "[Seniority-level gap]"],
  "overall_feedback": "Explicit profile/role fit assessment",
  "follow_up_question": "Question targeting resume/role alignment"
}}
'''

extract_knowledge_set_template = '''
list down top 5 core/important concepts from all the given skill in form of json  
{skills}

respond in json format:
{{
"skill1":[concept1,concept2,concept3 ...]
"skill2:[...]
}}
'''


# from string import Template
# template = Template("Hello, my name is $name and I live in $city. I work as a $job.")
# partial_filled = template.safe_substitute(name="Smit", city="Ahmedabad")
