check_language=f'''
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
- `rationale` (2-3 sentence explanation with evidence)  
- `quotes` (1-3 representative text excerpts illustrating the assessment)  

**Analysis Guidelines**  
| Feature          | Evaluation Criteria                                                                 |  
|------------------|-------------------------------------------------------------------------------------|  
| Clarity          | Sentence coherence, conciseness, ambiguity avoidance, filler usage                  |  
| Vocabulary       | Lexical diversity, context-appropriateness, sophistication, repetition analysis     |  
| Grammar & Syntax | Grammatical correctness, tense consistency, punctuation, sentence structure fluency |  
| Structure & Flow | Logical progression, paragraph transitions, thematic cohesion, argument sequencing |  
'''

extract_resume = f'''
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

