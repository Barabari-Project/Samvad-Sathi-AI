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
Year 0 Experience
{"category": "Technical", "difficulty": "Easy", "question": "Explain how the CSS Box Model works. What components make up an element's total width and height?", "hint": "Mention content, padding, border, margin and how they affect layout calculations"}
{"category": "Technical", "difficulty": "Medium", "question": "How would you implement a responsive grid layout without using frameworks like Bootstrap? Describe your approach.", "hint": "Use CSS Grid/Flexbox properties like grid-template-columns, fr units, and media queries"}
{"category": "Technical", "difficulty": "Hard", "question": "Walk through the browser's Critical Rendering Path. How would you optimize it for faster initial page loads?", "hint": "Cover DOM/CSSOM construction, render tree, layout, paint; mention async/defer, resource prioritization"}
{"category": "Behavioral", "difficulty": "Easy", "question": "Describe a time you had to learn a new technology quickly for a project. What steps did you take?", "hint": "Structure using STAR: Specific resource, Timeline, Application, Result"}
{"category": "Behavioral", "difficulty": "Medium", "question": "Tell me about a disagreement you had with a teammate regarding technical implementation. How was it resolved?", "hint": "Highlight active listening, objective criteria (e.g., performance metrics), and compromise"}
{"category": "Behavioral", "difficulty": "Hard", "question": "You discover a critical bug minutes before a deployment deadline. What actions do you take?", "hint": "Discuss triage, stakeholder communication, and tradeoffs between delay vs. risk"}
{"category": "Role-specific", "difficulty": "Easy", "question": "When would you choose React over vanilla JavaScript for a project? Provide examples.", "hint": "Compare component reusability, state management, and ecosystem tools"}
{"category": "Role-specific", "difficulty": "Medium", "question": "How would you structure a reusable accessibility-compliant button component in React?", "hint": "Include ARIA roles, keyboard events, focus management, and PropTypes"}
{"category": "Role-specific", "difficulty": "Hard", "question": "Design a real-time dashboard showing live analytics. Which frontend architecture patterns would you use?", "hint": "Discuss WebSocket integration, state normalization, throttling updates, and error fallbacks"}
{"category": "Resume-specific", "difficulty": "Easy", "question": "Your resume mentions an e-commerce dashboard project. What CSS techniques did you use for responsiveness?", "hint": "Reference specific methods like Flexbox breakpoints or relative units (em/rem)"}
{"category": "Resume-specific", "difficulty": "Medium", "question": "You listed TypeScript as a skill. How did it improve your Vue.js project at ABC Inc compared to plain JavaScript?", "hint": "Contrast type safety, refactoring ease, and error reduction with concrete examples"}
{"category": "Resume-specific", "difficulty": "Hard", "question": "Your resume states you improved load times by 30% at XYZ Corp. What specific frontend optimizations did you implement?", "hint": "Detail techniques like code splitting, lazy loading, or asset optimization tools"}
Year 2 Experience
{"category": "Technical", "difficulty": "Easy", "question": "What are React hooks? Provide examples of useState and useEffect.", "hint": "Explain side-effect management and state preservation across re-renders"}
{"category": "Technical", "difficulty": "Medium", "question": "Compare client-side rendering (CSR) vs. server-side rendering (SSR). When would you choose Next.js over plain React?", "hint": "Discuss SEO, TTI (Time to Interactive), and data-fetching tradeoffs"}
{"category": "Technical", "difficulty": "Hard", "question": "Debug a memory leak in a SPA. What tools and strategies would you use?", "hint": "Mention Chrome DevTools heap snapshots, useEffect cleanup, and event listener removal"}
{"category": "Behavioral", "difficulty": "Easy", "question": "How do you prioritize tasks when multiple features have tight deadlines?", "hint": "Reference frameworks like MoSCoW or effort/impact matrix"}
{"category": "Behavioral", "difficulty": "Medium", "question": "Describe mentoring a junior developer. How did you ensure their understanding of complex code?", "hint": "Include knowledge transfer methods (pair programming, docs) and feedback loops"}
{"category": "Behavioral", "difficulty": "Hard", "question": "You strongly disagree with an architectural decision from a senior engineer. How do you escalate your concerns?", "hint": "Emphasize data-driven arguments, prototypes, and respectful challenge"}
{"category": "Role-specific", "difficulty": "Easy", "question": "What metrics would you monitor to improve a website's Core Web Vitals?", "hint": "Name LCP, FID, CLS; explain how they affect user experience"}
{"category": "Role-specific", "difficulty": "Medium", "question": "Implement a dark/light theme switcher using CSS-in-JS. Handle persistence via localStorage.", "hint": "Use React context, ThemeProvider, and useEffect for syncing"}
{"category": "Role-specific", "difficulty": "Hard", "question": "Design an offline-first mobile PWA for a news site. How would you handle data synchronization?", "hint": "Cover service workers, cache strategies (NetworkFirst), and conflict resolution"}
{"category": "Resume-specific", "difficulty": "Easy", "question": "Your resume shows Express.js experience. How did you use it to build REST APIs for your e-commerce dashboard?", "hint": "Describe route handlers, middleware (e.g., CORS), and error handling"}
{"category": "Resume-specific", "difficulty": "Medium", "question": "You used WebSockets for real-time updates in your portfolio project. What challenges did you face with connection stability?", "hint": "Discuss heartbeat mechanisms, reconnection logic, and fallback options"}
{"category": "Resume-specific", "difficulty": "Hard", "question": "Your resume cites 'Redux for state management'. Why did you choose it over Context API for the XYZ Corp project?", "hint": "Compare middleware (thunk/saga), debugging tools, and performance at scale"}
Year 4 Experience
{"category": "Technical", "difficulty": "Easy", "question": "What is the virtual DOM? How does it differ from the actual DOM?", "hint": "Explain diffing algorithms and batched updates"}
{"category": "Technical", "difficulty": "Medium", "question": "Optimize a React app with heavy re-renders. What tools and techniques would you apply?", "hint": "Use React.memo, useCallback, React DevTools profiling, and code splitting"}
{"category": "Technical", "difficulty": "Hard", "question": "Design a micro-frontend architecture. How would you handle shared dependencies and cross-app communication?", "hint": "Discuss module federation, versioning, and event-bus patterns"}
{"category": "Behavioral", "difficulty": "Easy", "question": "How do you handle conflicting feedback from product and engineering teams on UI implementation?", "hint": "Describe facilitating workshops with prototypes to align goals"}
{"category": "Behavioral", "difficulty": "Medium", "question": "Share an example where you advocated for technical debt reduction. How did you justify the ROI?", "hint": "Quantify impact: e.g., 'Reduced build time by X%' or 'Fewer production incidents'"}
{"category": "Behavioral", "difficulty": "Hard", "question": "You’re leading a team through a major tech stack migration. How do you manage risk and ensure knowledge transfer?", "hint": "Cover incremental rollout, metrics tracking, and documentation rituals"}
{"category": "Role-specific", "difficulty": "Easy", "question": "When would you use GraphQL over REST for frontend-backend communication?", "hint": "Compare over-fetching, under-fetching, and schema flexibility"}
{"category": "Role-specific", "difficulty": "Medium", "question": "Implement a performant infinite scroll list. How do you avoid memory leaks?", "hint": "Use virtualization (e.g., react-window) and cleanup detached DOM nodes"}
{"category": "Role-specific", "difficulty": "Hard", "question": "Architect a design system supporting multiple products. How would you ensure consistency and theming?", "hint": "Discuss tokenization, component contracts, and monorepo tooling"}
{"category": "Resume-specific", "difficulty": "Easy", "question": "Your resume mentions Jest. How did you structure unit tests for your Vue.js components at ABC Inc?", "hint": "Reference mocking lifecycle hooks and testing emitted events"}
{"category": "Resume-specific", "difficulty": "Medium", "question": "You led a team for the e-commerce dashboard. How did you enforce code quality in a collaborative codebase?", "hint": "Detail PR review practices, linters (ESLint), and CI/CD checks"}
{"category": "Resume-specific", "difficulty": "Hard", "question": "Your resume shows a career gap in 2023. How did you use that time to upskill in frontend technologies?", "hint": "Highlight specific courses/certifications (e.g., advanced TypeScript) and personal projects"}
Year 6 Experience
{"category": "Technical", "difficulty": "Easy", "question": "Describe how React Fiber improves rendering performance.", "hint": "Explain incremental rendering and task prioritization"}
{"category": "Technical", "difficulty": "Medium", "question": "Evaluate strategies for bundle size reduction in a legacy Angular app. What tools would you use?", "hint": "Cover tree-shaking, lazy loading, and differential serving"}
{"category": "Technical", "difficulty": "Hard", "question": "Design a framework-agnostic component library. How would you handle cross-framework compatibility?", "hint": "Discuss Web Components, shadow DOM, and framework wrappers (e.g., React/Vue adapters)"}
{"category": "Behavioral", "difficulty": "Easy", "question": "How do you align frontend architecture decisions with long-term business goals?", "hint": "Link tech choices (e.g., SSR) to KPIs like conversion rates or SEO rankings"}
{"category": "Behavioral", "difficulty": "Medium", "question": "As a tech lead, how would you handle a team member consistently missing deadlines?", "hint": "Outline empathetic 1:1s, root-cause analysis, and mentorship plans"}
{"category": "Behavioral", "difficulty": "Hard", "question": "You’re asked to cut frontend team size by 30% while maintaining velocity. What’s your action plan?", "hint": "Focus on automation, removing low-impact features, and upskilling"}
{"category": "Role-specific", "difficulty": "Easy", "question": "Define BFF (Backend for Frontend) pattern. When is it beneficial?", "hint": "Describe aggregating microservices and tailoring data to UI needs"}
{"category": "Role-specific", "difficulty": "Medium", "question": "Implement A/B testing for a checkout flow rollout. How would you ensure statistical validity?", "hint": "Cover cohort isolation, metric selection (e.g., CVR), and significance testing"}
{"category": "Role-specific", "difficulty": "Hard", "question": "Propose a migration from REST to GraphQL for an enterprise-scale app. How would you phase it?", "hint": "Discuss schema stitching, gradual rollout, and client co-existence strategies"}
{"category": "Resume-specific", "difficulty": "Easy", "question": "Your resume lists Webpack optimizations. What loaders/plugins did you use for bundle splitting?", "hint": "Name specific tools like SplitChunksPlugin or MiniCssExtractPlugin"}
{"category": "Resume-specific", "difficulty": "Medium", "question": "You mention 'CI/CD pipelines' in your skills. How did you automate frontend testing/deployment at XYZ Corp?", "hint": "Detail stages like linting, visual regression tests, and canary releases"}
{"category": "Resume-specific", "difficulty": "Hard", "question": "Your resume shows 3 job changes in 4 years. How did each move advance your technical leadership capabilities?", "hint": "Connect role transitions to expanded scope (e.g., mentoring to architect)"}
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
    "category": "Technical",
    "difficulty": "Easy",
    "question": "Explain the difference between HTTP GET and POST methods.",
    "hint": "Safety and idempotency characteristics"
  },
  {
    "category": "Technical",
    "difficulty": "Medium",
    "question": "How would you optimize an API endpoint experiencing slow response times due to database queries?",
    "hint": "Caching strategies and query optimization techniques"
  },
  {
    "category": "Technical",
    "difficulty": "Hard",
    "question": "Design a distributed system for processing 1 million real-time payments per minute with guaranteed exactly-once semantics.",
    "hint": "Partitioning strategies and idempotency keys implementation"
  },
  {
    "category": "Behavioral",
    "difficulty": "Easy",
    "question": "Describe a situation where you received negative feedback on your code. How did you respond?",
    "hint": ""
  },
  {
    "category": "Behavioral",
    "difficulty": "Medium",
    "question": "Tell me about a technical decision you advocated for that was initially rejected by your team. How did you handle it?",
    "hint": "Evidence-based persuasion techniques"
  },
  {
    "category": "Behavioral",
    "difficulty": "Hard",
    "question": "Describe how you'd lead a team through a major system migration with zero downtime. Include how you'd handle dissenting opinions.",
    "hint": "Change management and phased rollout planning"
  },
  {
    "category": "Role-specific",
    "difficulty": "Easy",
    "question": "What factors would you consider when choosing between synchronous and asynchronous communication for microservices?",
    "hint": "Latency requirements and error handling"
  },
  {
    "category": "Role-specific",
    "difficulty": "Medium",
    "question": "How would you implement rate limiting in an API gateway for a fintech application?",
    "hint": "Token bucket vs leaky bucket algorithms"
  },
  {
    "category": "Role-specific",
    "difficulty": "Hard",
    "question": "Design a consensus mechanism for a distributed inventory management system during network partitions.",
    "hint": "Conflict-free replicated data types (CRDTs) application"
  },
  {
    "category": "Resume-specific",
    "difficulty": "Easy",
    "question": "Your resume mentions Kubernetes experience. Explain what a Deployment manages that a Pod doesn't.",
    "hint": "Replica management and rollout strategies"
  },
  {
    "category": "Resume-specific",
    "difficulty": "Medium",
    "question": "For your project 'Payment Gateway Integration', how did you ensure PCI compliance when storing credit card tokens?",
    "hint": "Tokenization standards and vault separation"
  },
  {
    "category": "Resume-specific",
    "difficulty": "Hard",
    "question": "During your 6-month gap in 2023, what specific technologies did you study and how have you applied them since?",
    "hint": "Concrete project applications of learned skills"
  },
  {
    "category": "Technical",
    "difficulty": "Medium",
    "question": "Compare eventual consistency vs strong consistency in distributed systems. When would you choose each?",
    "hint": "CAP theorem tradeoffs"
  },
  {
    "category": "Behavioral",
    "difficulty": "Medium",
    "question": "Describe a time you had to compromise technical quality to meet a deadline. What would you do differently now?",
    "hint": ""
  },
  {
    "category": "Role-specific",
    "difficulty": "Easy",
    "question": "What metrics would you monitor for a high-traffic messaging queue?",
    "hint": "Consumer lag and processing throughput"
  },
  {
    "category": "Resume-specific",
    "difficulty": "Hard",
    "question": "Your resume shows 3 companies in 4 years. Walk me through your career progression decisions during this period.",
    "hint": "Growth-oriented rationale for transitions"
  }
]
'''


backend_context = '''
'''


def get_gen_que_prompt(resume:str,YOE,JD,Role:str,NOQ:int):
   assert Role == "Data Science" or Role == "Frontend Developer" or Role == "Backend Developer"
   if JD:
         JD = "- Job Requirements: " + JD
   else:
         JD = ''
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



# print(get_gen_que_prompt(resume="# RESUME #",YOE="# YEARS_OF_Experience #",Role="Frontend Developer",NOQ=0,JD=''))