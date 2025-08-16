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
- `rationale` (2-3 sentence explanation using SIMPLE ENGLISH with clear evidence)  
- `quotes` (1-3 complete context excerpts showing full phrases/sentences)  

**Analysis Guidelines**  
| Feature          | Evaluation Criteria                                                                 |  
|------------------|-------------------------------------------------------------------------------------|  
| Clarity          | Sentence coherence, conciseness, ambiguity avoidance, filler usage                  |  
| Vocabulary       | Lexical diversity, context-appropriateness, sophistication, repetition analysis     |  
| Grammar & Syntax | Grammatical correctness, tense consistency, punctuation, sentence structure fluency |  
| Structure & Flow | Logical progression, paragraph transitions, thematic cohesion, argument sequencing |  

**Critical Instructions**  
1. Use ONLY simple, easy-to-understand English in rationales  
2. Quotes MUST be self-contained with full context (complete clauses/sentences)  
3. Strictly maintain original JSON structure and keys  
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

analyse_domain_template = '''
**Role**  
You are an expert `{job_title}`. Your task is to rigorously evaluate a candidate's domain knowledge based on their response to an interview question.

---

### Instructions  
1. **Category-Driven Analysis**  
   - Analyze *only* the attributes relevant to the question's category:  
     - **Technical**: Focus on Accuracy, Depth, Terminology, Examples.  
     - **Behavioral**: Focus on Examples/Evidence, Depth, Relevance. *Ignore Terminology Usage*.  
     - **Role-specific**: All attributes EXCEPT Terminology *unless* the question demands domain jargon.  
     - **Resume-specific**: Prioritize Examples/Evidence and Accuracy (validate against `Users_Resume_Profile`).  

2. **Hint Handling**  
   - If a hint exists, reward responses that explicitly follow it.  
   - Example: A hint like "STAR method" expects Situation/Task/Action/Result structure.  

3. **Scoring (1-5 per Attribute)**  
   - **Accuracy**: Factual correctness.  
     - *5: Flawless, 3: Partially correct, 1: Incorrect*  
   - **Depth of Understanding**: Nuance/complexity.  
     - *5: Detailed tradeoffs, 3: Surface-level, 1: Vague*  
   - **Relevance**: Addresses all question parts.  
     - *5: Fully on-point, 3: Partial, 1: Off-topic*  
   - **Examples/Evidence**: Concrete proof.  
     - *5: Specific case studies, 3: Generic, 1: None*  
   - **Terminology Usage** (Technical/Role-specific only):  
     - *5: Precise jargon, 3: Minor errors, 1: Misused terms*  

4. **Resume Validation**  
   - For Resume-specific questions, cross-check claims against `Users_Resume_Profile`. Flag inconsistencies.  

---

### Examples of Analysis  
#### Example 1: Technical Question (0 YOE)  
**Question**:  
*"Explain bias-variance tradeoff using a simple linear regression example. Hint: Visualize underfitting vs overfitting curves."*  
**Response**:  
*"High bias (underfitting) occurs when linear regression oversimplifies data. High variance (overfitting) happens with complex models memorizing noise."*  
**Analysis**:  
- **Accuracy**: 3/5 (Correct basics but misses linear regression example).  
- **Depth**: 2/5 (No tradeoff mechanics or complexity impact).  
- **Relevance**: 4/5 (Addresses core concepts).  
- **Examples**: 1/5 (No regression example/visualization).  
- **Terminology**: 5/5 (Correct terms).  
- **Hint Followed?** No → Penalized Depth/Examples.  

#### Example 2: Behavioral Question (4 YOE)  
**Question**:  
*"Describe resolving model fairness issues. Hint: Equal odds vs opportunity parity."*  
**Response**:  
*"We prioritized equal opportunity by adjusting thresholds for loan approvals, reducing false negatives in protected groups."*  
**Analysis**:  
- **Depth**: 5/5 (Nuanced fairness tradeoffs).  
- **Examples**: 5/5 (Specific threshold strategy).  
- **Relevance**: 5/5 (Uses hint's "opportunity" focus).  
- *Terminology Skipped* (Behavioral category).  

---

### Context  
- **Job Title**: `{job_title}`  
- **Expected Seniority**: `{Years_of_experience}` years (e.g., Junior/Senior).  
- **Users Resume Profile's **:  
  ```  
  {Users_Resume_Profile}  
  ```  
- **Question Metadata**:  
  - Category: `{category}`  
  - Difficulty: `{difficulty}`  
  - Hint: `{hint}`  

---

### Analysis Task  
**Interview Question**:  
"{question}"  

**Candidate Response**:  
"{Candidate_Response}"  

**Your Analysis**:  
**Return your analysis strictly in the following JSON format:**
{{
  "category": "[category]",
  "hint_addressed": true/false/null, // null if no hint
  "attribute_scores": {{
    "Accuracy": {{"score": number, "reason": string}},
    "Depth of Understanding": {{"score": number, "reason": string}},
    "Relevance": {{"score": number, "reason": string}},
    "Examples/Evidence": {{"score": number, "reason": string}},
    "Terminology Usage": {{"score": number, "reason": string}} // Omit if behavioral
  }},
  "overall_score": number, // sum of all score
  "overall_feedback": "Concise strengths/weaknesses summary",
  "actionable_feedback": "- Fix X\n- Do Y\n- Try Z" 
}}

---

### Rules  
- **No assumptions**: Base scores strictly on the response.  
- **Hint bonus**: If a hint exists, +0.5 to Accuracy/Depth when followed (round down if .49).  
- **Resume checks**: For Resume-specific questions, deduct -1 from Accuracy if claims contradict.  
- **Be brutal**: Ignore fluff. Reward precision. Penalize vagueness.
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

  
Final_Summary_template = """
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