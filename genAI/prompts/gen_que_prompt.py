prompt_template = """
**ROLE**
You are an expert interviewer with deep experience in hiring for the role of {Role}. Your task is to create highly tailored, insightful interview questions based on the candidate's resume, work history, skills, and years of experience.

### INSTRUCTION
1. **Question Generation**
   - Create **15-20 questions** covering the following categories:
     - **Technical (60%)**: Core DS concepts such as machine learning, statistics, and coding. These should assess depth of understanding and problem-solving abilities.
     - **Behavioral (20%)**: Soft skills including teamwork, communication, leadership, and navigating challenges.
     - **Role-specific (10%)**: Alignment with the specific job responsibilities and domain expertise required for the role (e.g., NLP, MLOps, GenAI).
     - **Resume-specific (10%)**: Directly derived from the candidate's resume — such as past projects, tools used, accomplishments, or any apparent career gaps.
   - **Label each question** with its category (e.g., `[Technical]`).
   - Elaborate on each question to provide clarity and context for the candidate.
   
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
     - Projects listed
     - Skills claimed (e.g., "Python," "Express")
     - Experience gaps/job hops

4. **Output Format**
   {{
      "category": "Technical/Behavioral/Role-specific/Resume-specific",
      "difficulty": "Easy/Medium/Hard",
      "question": "Your question here",
      "hint":"hint referaning to question"
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
### 1. **0 Years Experience** (Entry-level)
[
  {
    "category": "Technical",
    "difficulty": "Easy",
    "question": "Explain bias-variance tradeoff using a simple linear regression example. How does model complexity affect this balance?",
    "hint": "Visualize underfitting vs overfitting curves"
  },
  {
    "category": "Technical",
    "difficulty": "Easy",
    "question": "Write Python code to handle missing values in a Pandas DataFrame using three different imputation strategies.",
    "hint": "Consider mean, median, and interpolation"
  },
  {
    "category": "Behavioral",
    "difficulty": "Medium",
    "question": "Describe a time you explained technical concepts to non-technical stakeholders during your academic projects.",
    "hint": "STAR method with specific project reference"
  },
  {
    "category": "Resume-specific",
    "difficulty": "Easy",
    "question": "Your capstone project mentions 'customer segmentation using K-means'. How did you determine optimal cluster count?",
    "hint": "Expect elbow method or silhouette score"
  }
]

### 2. **2 Years Experience** (Junior)
```json
[
  {
    "category": "Technical",
    "difficulty": "Medium",
    "question": "When deploying a model with Sklearn and Flask, how would you ensure consistent preprocessing between training and inference?",
    "hint": "Pipeline serialization"
  },
  {
    "category": "Role-specific",
    "difficulty": "Hard",
    "question": "Design an A/B testing framework for email campaign conversion rates. What statistical tests would you use at 500k samples?",
    "hint": "Z-test vs t-test considerations"
  },
  {
    "category": "Resume-specific",
    "difficulty": "Medium",
    "question": "Your resume shows a 3-month gap after your first job. What skills did you acquire during this period relevant to ML engineering?",
    "hint": "Expect specific tool/library names"
  }
]
```

### 3. **4 Years Experience** (Mid-level)
[
  {
    "category": "Technical",
    "difficulty": "Hard",
    "question": "How would you implement zero-downtime deployment for a real-time fraud detection model? Include canary release strategy.",
    "hint": "Shadow mode testing"
  },
  {
    "category": "Behavioral",
    "difficulty": "Hard",
    "question": "Describe resolving model fairness issues where accuracy tradeoffs disadvantaged protected groups. What metrics did you prioritize?",
    "hint": "Equal odds vs opportunity parity"
  },
  {
    "category": "Role-specific",
    "difficulty": "Medium",
    "question": "For time-series forecasting in inventory management, how would you handle sudden demand spikes during holidays?",
    "hint": "Exogenous variable inclusion"
  }
]

### 4. **6 Years Experience** (Senior)
[
  {
    "category": "Technical",
    "difficulty": "Hard",
    "question": "Design a cost-optimized LLM serving architecture for 10k RPM with <100ms latency. Compare monolithic vs microservice approaches.",
    "hint": "Autoscaling with spot instances"
  },
  {
    "category": "Behavioral",
    "difficulty": "Hard",
    "question": "When have you overruled technical debt concerns to meet business deadlines? What was the long-term impact?",
    "hint": "Quantified tech debt consequences"
  },
  {
    "category": "Resume-specific",
    "difficulty": "Hard",
    "question": "Your MLOps platform reduced deployment time by 40%. What specific bottlenecks did you eliminate in the CI/CD pipeline?",
    "hint": "Artifact reproducibility or testing"
  }
]
'''

ML_context = '''
Questions must draw **exclusively** from these topics:

### **Enhanced Technical Syllabus**  
**1. Statistics & Probability**  
- *Hypothesis Testing*: T-tests, Z-tests, ANOVA, p-values, Type I/II errors  
- *Bayesian Inference*: Priors/posteriors, Bayes' theorem applications  
- *Distributions*: Gaussian, Poisson, Binomial properties and use cases  
- *A/B Testing*: Power analysis, sequential testing, covariate adjustment  

**2. Machine Learning**  
- *Supervised Learning*:  
  - Algorithms: Linear/logistic regression, SVM, tree-based methods (RF, XGBoost)  
  - Evaluation: ROC curves, precision-recall tradeoffs, cross-validation strategies  
- *Unsupervised Learning*: K-means clustering, PCA, anomaly detection  
- *Regularization*: L1/L2 penalties, dropout, early stopping  

**3. Coding & Tools**  
- *Python*:  
  - Pandas (data wrangling, time-series manipulation)  
  - Scikit-learn (pipeline construction, hyperparameter tuning)  
- *SQL*: Window functions, query optimization, nested queries  
- *Git*: Branching strategies, rebase vs. merge, CI/CD integration  
- *Cloud Platforms*:  
  - AWS (SageMaker, Redshift) / GCP (BigQuery, Vertex AI)  

**4. Data Engineering**  
- *ETL Pipelines*: Batch vs. stream processing (Airflow vs. Kafka)  
- *Data Warehousing*: Star/snowflake schemas, slowly changing dimensions  
- *Big Data Tools*: Spark (RDD/DataFrame API), Hadoop ecosystem  

**5. Advanced Topics**  
- *Deep Learning*:  
  - CNNs (architectures like ResNet, transfer learning)  
  - RNNs (LSTM/GRU, sequence-to-sequence models)  
- *NLP*: Transformer architecture, fine-tuning BERT, attention mechanisms  
- *MLOps*: Model monitoring, drift detection, feature stores  

---

### **Behavioral Syllabus**  
**Core Framework**  
- **STAR Method**: Structured storytelling (Situation, Task, Action, Result) with quantifiable outcomes  
- **Prioritization**: Eisenhower matrix, ROI-based task triage, resource constraints  
- **Ethical Dilemmas**: Data privacy (GDPR/CCPA), model bias mitigation, explainability tradeoffs  
- **Stakeholder Communication**: Tailoring messages to executives vs. engineers, conflict resolution  

**Integrated Amazon Leadership Principles**  
1. **Earn Trust**:  
   - *Focus*: Building psychological safety, admitting mistakes, delivering on commitments  
   - *Sample Q*: "Describe a time you received critical feedback. How did you rebuild trust?"  

2. **Are Right, A Lot**:  
   - *Focus*: Data-driven decision-making, balancing intuition with evidence, handling ambiguous data  
   - *Sample Q*: "When did you advocate for a counterintuitive solution backed by data?"  

3. **Invent and Simplify**:  
   - *Focus*: Creating scalable solutions, reducing technical debt, elegant problem-solving  
   - *Sample Q*: "Share an example where you turned a complex process into a simple solution."  

### **Role-Specific Additions**  
**NLP Specialist**  
- *Must-Know*: Attention mechanisms, transformer variants (RoBERTa, T5), Hugging Face ecosystem  
- *Tools*: spaCy, NLTK, BERTopic  

**ML Engineer**  
- *Must-Know*: Containerization (Docker), serverless deployment, model versioning (MLflow)  
- *Tools*: Kubernetes, TF Serving, Prometheus for monitoring  

**Analytics Lead**  
- *Must-Know*: Causal inference (propensity scoring, DiD), cohort analysis, monetization metrics  
- *Tools*: Optimizely, Mixpanel, Monte Carlo simulations  
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


def get_gen_que_prompt(resume:str,YOE:int,JD,Role:str,NOQ:int):
   assert Role == "Data Science" or Role == "Frontend Developer" or Role == "Backend Developer"
   if JD:
         JD = "- Job Requirements: " + JD
   else:
         JD = ''
   example,context = "",""
   if Role == "Data Science":
      example = "$Examples" # ML_examples
      context = ML_context
      resume = "$resume"
      YOE = "$X years of experience"
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

ml_resume = '''
{
  "experience": [
    {
      "company": "The Barabari Collective",
      "position": "AI Engineer (Freelance)",
      "duration": "June 2025 - Present"
    },
    {
      "company": "GDSC DDU Chapter",
      "position": "AI/ML Team Member",
      "duration": "2023-2024"
    }
  ],
  "certifications": [
    "Advanced Learning Algorithms - DeepLearning.AI",
    "Introduction to TensorFlow for AI, ML and DL - DeepLearning.AI",
    "Supervised Machine Learning - Stanford University"
  ],
  "projects": [
    {
      "name": "ChatGPT 2",
      "description": "Trained and implemented from scratch in Pytorch. Winner of Bhashathon 2025, won a cash prize of Rs. 50,000."
    }
  ],
  "skills": {
    "Languages": [
      "Python",
      "C++",
      "JavaScript"
    ],
    "Libraries": [
      "PyTorch",
      "NumPy",
      "Pandas",
      "Matplotlib",
      "FAISS",
      "scikit-learn"
    ],
    "Frameworks": [
      "Flask",
      "FastAPI",
      "Express.js",
      "TensorFlow"
    ],
    "Tools & Technologies": [
      "Git",
      "AWS",
      "Linux",
      "SentencePiece",
      "OpenAI",
      "OpenAI-Agents",
      "Deepgram",
      "openSMILE"
    ]
  }
}
'''
print(get_gen_que_prompt(resume=ml_resume,YOE=0,JD='',Role="Data Science",NOQ=10))