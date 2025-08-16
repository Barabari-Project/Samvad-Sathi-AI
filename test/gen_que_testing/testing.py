
import json
from prompts import analyse_domain_template as orignial
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import pandas as pd
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from tqdm import tqdm

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
        start = int(start)
        end = int(end)
        json_str = text[start:end]
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(text)
        print(json_str)
        raise ValueError(f"Invalid JSON found: {e}")
      
class llm_with_retry:
  def __init__(self):
      self.itr = 0
    
  def call(self,prompt):
    try:
      res = call_llm(prompt=prompt,model="gpt-4o")
      res = extract_json_dict(res)
      self.itr = 0
      return res
    except:
      if self.itr<3:
        self.itr += 1
        print("RETRY: ",self.itr)
        return self.call(prompt=prompt)
      else:
        self.itr=0
        return -1

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

### Rules  
- **No assumptions**: Base scores strictly on the response.  
- **Hint bonus**: If a hint exists, +0.5 to Accuracy/Depth when followed (round down if .49).  
- **Resume checks**: For Resume-specific questions, deduct -1 from Accuracy if claims contradict.  
- **Be brutal**: Ignore fluff. Reward precision. Penalize vagueness.
'''

# ['backend_ques.json','data_science.json','frontend_ques.json']
with open('data_science.json', 'r') as file:
    ques = json.load(file)

dataset = {
    "Question": [],
    "varient": [],
    "answer": [],
    "accuracy": [],
    "depth": [],
    "relevance": [],
    "hint": [],
    "accuracy_reason": [],
    "relevance_reason": [],
    "depth_reason": [],
    "terminology": [],
    "terminology_reason": [],
    "Examples/Evidence": [],
    "Examples/Evidence_reason": []
}

print(len(ques))
QID = 0
llm = llm_with_retry()
for que in ques:
  category, difficulty, question, hint = que.values()
  QID+=1
  print('QID :',QID)
  job_title = 'data_science'
  YOE = '# YOE #'
  Resume = '# Resume #'
  prompt = analyse_domain_template.format(
    category=category, difficulty=difficulty, question=question, hint=hint,job_title=job_title,Years_of_experience=YOE,Users_Resume_Profile=Resume,Candidate_Response="# ANSWER #"
  )
  
  cases = ['''  
  Addresses the full question directly.
  Uses hint properly (e.g., STAR format, visualization, or keywords).
  Gives precise, correct details with technical terminology.
  Provides nuanced reasoning and examples.
  ''',
  
  '''
  Follows the hint format mechanically (e.g., gives STAR paragraphs).
  But: information is incorrect, off-topic, or vague.
  No real insight or factual value.
  Examples may be generic or fabricated.
  ''',
  
  '''
  Fully answers the question with precision, clarity, and deep insight.
  But completely ignores the hint, which may ask for STAR or specific concepts.
  Gives specific examples and valid reasoning.
  ''',
  
  '''
  Attempts to follow the hint but misuses it or drifts from its intent.
  Has thoughtful reasoning and tradeoffs.
  Factual inaccuracies (e.g., misapplies a theory or uses wrong terminology).
  Possibly relevant but technically incorrect.
  ''',
  
  '''
  Reasonable answer, mostly accurate.
  Follows the hint structure well.
  Lacks technical richness or example quality.
  May have some vagueness or minor errors.
  ''']
  
  Table = '''
  | # | Hint | Accuracy | Relevance | Depth  | Type                      | Score Range |
  | - | ---- | -------- | --------- | ------ | ------------------------- | ----------- |
  | 1 | ✅    | High     | High      | High   | ⭐ Perfect                 | 24-25       |
  | 2 | ✅    | Low      | Low       | Low    | ❌ Failure                 | 6-8         |
  | 3 | ❌    | High     | High      | High   | ⚠️ Smart but Ignored Hint | 21-22       |
  | 4 | ⚠️   | Low      | High      | High   | 🤔 Deep but Flawed        | 12-15       |
  | 5 | ✅    | Medium   | High      | Medium | 🟡 Realistic Midpoint     | 17-19       |
  '''
  
  vars = ['''Follows hint, has high accuracy,is very relevent,  answer is in depth''',
        "Follows hint, has low accuracy, is not relevant, answer lacks depth",
        "Ignores hint, has high accuracy, is very relevant, answer is in depth",
        "Partially follows hint, has low accuracy, is highly relevant, answer is in depth",
        "Follows hint, has medium accuracy, is highly relevant, answer is moderately in depth"]
  
  vars_score = [[1,5,5,5],[1,1,1,1],[0,5,5,5],[0.5,1,5,5],[1,2.5,5,2.5]]
  
  add_on = '''
  Your task is to genarate answer with strictly following these charecteristics for the given interview question.
  {case}
  
  Question should be answered such that it should have score relative to [{var}]
  
  Question : 
  {que}
  
  this is how the answer would be evaluated:
  {prompt}
  
  Make sure to strcitly response in json format with only one key "answer"
  For example:
  {{
    "answer" : str  
  }}
  '''
  from tqdm import tqdm
  import pandas as pd

  # Safe nested get function
  def deep_get(dictionary, keys, default=None):
      for key in keys:
          if isinstance(dictionary, dict):
              dictionary = dictionary.get(key, default)
          else:
              return default
      return dictionary

  for ix, (case, var, exp_scores) in enumerate(tqdm(zip(cases, vars, vars_score), total=len(cases))):
      retries = 0
      while retries < 3:
          try:
              final = add_on.format(case=case, var=var, prompt=prompt, que=que)
              res = llm.call(final)
              if isinstance(res, int):
                  pd.DataFrame(dataset).to_csv("dataset.csv")
                  print("Error in generating answer")
                  exit(0)

              ans = res.get("answer", "")

              exp_hint, exp_accuracy, exp_relevent, exp_depth = exp_scores

              prompt = orignial.format(
                  category=category,
                  difficulty=difficulty,
                  question=question,
                  hint=hint,
                  job_title=job_title,
                  Years_of_experience=YOE,
                  Users_Resume_Profile=Resume,
                  Candidate_Response=ans
              )

              res = llm.call(prompt)
              if isinstance(res, int):
                  pd.DataFrame(dataset).to_csv("dataset.csv")
                  print("Error in generating analysis")
                  exit(0)

              gt_hint = res.get('hint_addressed', '')
              gt_accuracy = deep_get(res, ['attribute_scores', 'Accuracy', 'score'], 0)
              gt_relevent = deep_get(res, ['attribute_scores', 'Relevance', 'score'], 0)
              gt_depth = deep_get(res, ['attribute_scores', 'Depth of Understanding', 'score'], 0)

              dataset["Question"].append(str(QID) + '. ' + str(que))
              dataset["varient"].append(ix)
              dataset["answer"].append(ans)
              dataset["hint"].append(str(gt_hint)+'/'+str(exp_hint))
              dataset["accuracy"].append(str(gt_accuracy)+'/'+str(exp_accuracy))
              dataset["relevance"].append(str(gt_relevent)+'/'+str(exp_relevent))
              dataset["depth"].append(str(gt_depth)+'/'+str(exp_depth))

              dataset["accuracy_reason"].append(deep_get(res, ['attribute_scores', 'Accuracy', 'reason'], ''))
              dataset["relevance_reason"].append(deep_get(res, ['attribute_scores', 'Relevance', 'reason'], ''))
              dataset["depth_reason"].append(deep_get(res, ['attribute_scores', 'Depth of Understanding', 'reason'], ''))
              dataset["terminology"].append(deep_get(res, ['attribute_scores', 'Terminology Usage', 'score'], 0))
              dataset["terminology_reason"].append(deep_get(res, ['attribute_scores', 'Terminology Usage', 'reason'], ''))
              dataset["Examples/Evidence"].append(deep_get(res, ['attribute_scores', 'Examples/Evidence', 'score'], 0))
              dataset["Examples/Evidence_reason"].append(deep_get(res, ['attribute_scores', 'Examples/Evidence', 'reason'], ''))

              break  # Exit retry loop on success

          except Exception as e:
              retries += 1
              pd.DataFrame(dataset).to_csv("dataset.csv")
              print("Error on QID", QID, "| Retry:", retries, "| Error:", e)

        # exit(0)
    
  # break
dataset = pd.DataFrame(dataset)
dataset.to_csv("dataset.csv")