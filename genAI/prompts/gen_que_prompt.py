prompt_template = """
**ROLE**
You are an expert interviewer with deep experience in hiring for the role of {Role}. Your task is to create highly tailored, insightful interview questions based on the candidate's resume, work history, skills, and years of experience.

### INSTRUCTION
1. **Question Generation**
   - Create **15-20 questions** covering these categories, elaborate on questions to help the candidate understand the question better:
     - **Technical**: Core DS concepts (ML, stats, coding)
     - **Behavioral**: Soft skills, teamwork, problem-solving
     - **Role-specific**: Job description alignment (e.g., NLP, MLOps)
     - **Resume-specific**: Directly from resume content (projects, skills, gaps)
   - **Label each question** with its category (e.g., `[Technical]`).

2. **Difficulty Settings** (based on years of experience):
   | **Experience** | **Easy** | **Medium** | **Hard** | **Distribution** |
   |----------------|----------|------------|----------|-----------------------|
   | **0-2 years** | Foundational concepts | Basic applications | Theoretical depth | 50% Easy, 30% Medium, 20% Hard |
   | **2-5 years** | Applied scenarios | Optimization trade-offs | System design | 30% Easy, 40% Medium, 30% Hard |
   | **5+ years** | Edge cases | Architecture decisions | Leadership/strategy | 20% Easy, 30% Medium, 50% Hard |

   - **Difficulty Definitions**:
     - **Easy**: Definitions, basic syntax, simple scenarios.
     - **Medium**: Applied problem-solving, trade-off analysis.
     - **Hard**: System design, optimization, failure mitigation.

3. **Resume Integration**
   - Extract **3 resume-specific questions** targeting:
     - Projects listed
     - Skills claimed (e.g., "Python," "Express")
     - Experience gaps/job hops

4. **Output Format**
   {{
      "category": "Technical/Behavioral/Role-specific/Resume-specific",
      "difficulty": "Easy/Medium/Hard",
      "question": "Your question here"
   }}
   - **Do NOT** add explanations or numbering.
   
### EXAMPLE
{examples}

### CONTEXT (Knowledge Grounding / Syllabus)
Questions must draw exclusively from these topics:
{context}

### TASK
Generate questions for:
- **Years of Experience**: {years_of_experience}
- **Resume**: {user_resume}
{job_description}

**Output**: 15-20 questions following the format above.

### KEY RULES
1. **No open-ended decisions**: All questions MUST adhere to the syllabus and difficulty rules.
2. **Resume fidelity**: Resume-specific questions must reference exact projects/skills.
3. **Diversity**: Cover ≥2 topics per category (e.g., stats + ML for Technical).
4. **Role alignment**: 40% of questions must map to the job description.
"""

ML_examples = '''
[
  {
    "category": "Technical",
    "difficulty": "Medium",
    "question": "Explain how you'd mitigate overfitting in a CNN for image classification."
  },
  {
    "category": "Behavioral",
    "difficulty": "Easy",
    "question": "Describe a time you resolved a conflict in a cross-functional team."
  },
  {
    "category": "Role-specific",
    "difficulty": "Hard",
    "question": "Design a real-time fraud detection system for transactional data (assume 1M RPM)."
  },
  {
    "category": "Resume-specific",
    "difficulty": "Medium",
    "question": "Your resume mentions a CNN project—what data augmentation techniques did you use and why?"
  }
]
'''

ML_context = '''
Questions must draw **exclusively** from these topics:

#### **Technical Syllabus**
1. **Statistics & Probability**
   - Hypothesis testing, Bayesian inference, distributions, A/B testing
2. **Machine Learning**
   - Supervised/unsupervised learning, evaluation metrics (AUC, F1), regularization
3. **Coding & Tools**
   - Python (Pandas, Scikit-learn), SQL, Git, cloud platforms (AWS/GCP)
4. **Data Engineering**
   - ETL pipelines, data warehousing, Spark/Kafka
5. **Advanced Topics**
   - Deep Learning (CNNs, RNNs), NLP (BERT, transformers), MLOps

#### **Behavioral Syllabus**
   - STAR method, prioritization, ethical dilemmas, stakeholder communication

#### **Role-specific Syllabus**
   - **NLP Specialist**: Topic modeling, transformers
   - **ML Engineer**: Model deployment, CI/CD
   - **Analytics Lead**: Experiment design, business metrics
'''

frontend_examples = '''
[
  {
    "candidate_summary": "1 year of experience, resume includes 'Mobile-first PWA for local business.'",
    "questions": [
      {
        "category": "Technical",
        "difficulty": "Easy",
        "question": "When would you use CSS Grid vs. Flexbox? Provide layout examples."
      },
      {
        "category": "Behavioral",
        "difficulty": "Medium",
        "question": "How would you handle disagreements with a designer about implementation feasibility?"
      },
      {
        "category": "Role-specific",
        "difficulty": "Medium",
        "question": "Design a service worker caching strategy for a PWA with frequently updated product catalogs."
      },
      {
        "category": "Resume-specific",
        "difficulty": "Hard",
        "question": "Your PWA claims 40% faster TTI—what metrics did you track and how did you achieve this?"
      }
    ]
  },
  {
    "candidate_summary": "6 years of experience, resume includes 'Micro-frontend architecture migration.'",
    "questions": [
      {
        "category": "Technical",
        "difficulty": "Hard",
        "question": "Compare hydration strategies for SSR applications (e.g., React vs Qwik)."
      },
      {
        "category": "Behavioral",
        "difficulty": "Hard",
        "question": "Describe a technical debt situation you inherited and how you drove systemic fixes."
      },
      {
        "category": "Role-specific",
        "difficulty": "Medium",
        "question": "How would you implement a design system to ensure consistency across micro-frontends?"
      },
      {
        "category": "Resume-specific",
        "difficulty": "Medium",
        "question": "Your migration project mentions Webpack Module Federation—what were the biggest integration challenges?"
      }
    ]
  }
]
'''

frontend_context = """
#### **Technical Syllabus**  
1. **Core Web Technologies**  
   - HTML5 (semantic elements, accessibility), CSS3 (Flexbox/Grid, animations), JavaScript (ES6+, async/event loop)  
   - DOM manipulation, browser APIs, Web Performance (Critical Rendering Path, Lighthouse)  

2. **Frontend Frameworks & Libraries**  
   - React (hooks, state management), Vue, Angular, Svelte  
   - Component lifecycle, virtual DOM, SSR/SSG (Next.js, Nuxt)  

3. **State Management & Data Handling**  
   - Redux, Context API, React Query, GraphQL/Apollo  
   - RESTful APIs, WebSockets, error handling  

4. **Styling & Design Systems**  
   - CSS-in-JS (Styled Components, Emotion), preprocessors (Sass)  
   - Responsive design, cross-browser compatibility, UI/UX principles  

5. **Tooling & Workflow**  
   - Build tools (Webpack, Vite), testing (Jest, React Testing Library, Cypress)  
   - CI/CD pipelines, package managers (npm/yarn), TypeScript  

#### **Behavioral Syllabus**
   - STAR method, prioritization, ethical dilemmas, stakeholder communication

#### **Role-specific Syllabus**  
   - **UX-Focused Frontend**: Component libraries, interaction design  
   - **Performance Specialist**: Bundle optimization, Core Web Vitals  
   - **Full-Stack Frontend**: API integration, BFF (Backend For Frontend) patterns  
   - **Accessibility Engineer**: WCAG compliance, ARIA, assistive tech testing  
"""

backend_examples = '''
[
  {
    "candidate": "3 years backend experience, resume highlights 'Scaled payment API handling 5K RPM'",
    "questions": [
      {
        "category": "Technical",
        "difficulty": "Medium",
        "question": "How would you implement idempotency in a payment processing API?"
      },
      {
        "category": "Behavioral",
        "difficulty": "Medium",
        "question": "Tell me about a time you had to refactor critical code under tight deadlines. What trade-offs did you consider?"
      },
      {
        "category": "Role-specific",
        "difficulty": "Hard",
        "question": "Design a rate-limiting system for our API that supports 100K+ unique clients with dynamic throttling rules."
      },
      {
        "category": "Resume-specific",
        "difficulty": "Medium",
        "question": "Your payment API handled 5K RPM - what strategies did you use for database connection pooling under load?"
      }
    ]
  },
  {
    "candidate": "1 year experience, resume includes 'Deployed serverless microservices on AWS'",
    "questions": [
      {
        "category": "Technical",
        "difficulty": "Easy",
        "question": "Explain how you'd choose between SQS and Kafka for inter-service communication."
      },
      {
        "category": "Behavioral",
        "difficulty": "Easy",
        "question": "Describe a technical challenge you faced during an internship and how you sought help."
      },
      {
        "category": "Role-specific",
        "difficulty": "Medium",
        "question": "How would you design a file processing pipeline where uploads take >15 minutes? Consider cost and reliability."
      },
      {
        "category": "Resume-specific",
        "difficulty": "Medium",
        "question": "For your serverless project, what monitoring metrics did you track and why?"
      }
    ]
  }
]
'''


backend_context = '''
'''


def get_gen_que_prompt(resume:str,YOE:int,JD:int,Role:str,NOQ:int):
  assert Role == "Data Science" or Role == "Frontend Developer" or Role == "Backend Developer"
  example,context = "",""
  if Role == "Data Science":
    example = ML_examples
    context = ML_context
  elif Role == "Frontend Developer":
    example = frontend_examples
    context = frontend_context
  elif Role == "Backend Developer":
    example = backend_examples
    context = backend_context
  
  prompt = prompt_template.format(user_resume=resume,
                                  years_of_experience=YOE,
                                  job_description=JD,
                                  examples=example,
                                  context=context,
                                  Role=Role,
                                  )
  return prompt
