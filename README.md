# HR Navigator — Workforce Intelligence Platform

HR Navigator is an AI-powered HR analytics platform that helps you understand the full employee lifecycle — from hiring to retention and promotions. It combines machine learning models, interactive dashboards, and simple resume screening tools into one unified system built with Streamlit.

---

## Project structure

```
hr-navigator/
│
├── app.py                      Main entry point (run this file)
│
├── _pages/
│   ├── home.py                 Landing page and employee journey overview
│   ├── recruitment.py          Hiring prediction + resume screening
│   ├── retention.py            Attrition prediction + risk dashboard
│   ├── promotion.py            Promotion eligibility and readiness scoring
│   └── analytics.py            Overall HR insights dashboard
│
├── .streamlit/
│   └── config.toml             Theme and configuration
│
├── requirements.txt            Project dependencies
├── .gitignore
└── README.md
```

---

## How to run locally

### Step 1: Check Python version

Make sure you have Python 3.10 or higher installed.

```bash
python --version
```

If not, download it from [https://www.python.org/downloads/](https://www.python.org/downloads/)

---

### Step 2: Clone the project

```bash
git clone https://github.com/YOUR_USERNAME/hr-navigator.git
cd hr-navigator
```

Or download the ZIP and open the folder in a terminal.

---

### Step 3: Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

---

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 5: Run the app

```bash
streamlit run app.py
```

Then open:
[http://localhost:8501](http://localhost:8501)

---

## Deploy to Streamlit Cloud (free hosting)

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/hr-navigator.git
git push -u origin main
```

---

### Step 2: Deploy

1. Go to [https://share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click “New app”
4. Select:

   * Repository: hr-navigator
   * Branch: main
   * Main file: app.py
5. Click Deploy

Your app will be live in a few minutes.

---

## Datasets used

### Recruitment

Used for hiring prediction and candidate evaluation.

Columns:
Age, Gender, EducationLevel, ExperienceYears, PreviousCompanies, DistanceFromCompany, InterviewScore, SkillScore, RecruitmentStrategy, HiringDecision, PersonalityScore

Dataset:
[https://www.kaggle.com/datasets/rabieelkharoua/predicting-hiring-decisions-in-recruitment-data](https://www.kaggle.com/datasets/rabieelkharoua/predicting-hiring-decisions-in-recruitment-data)

---

### Retention

Used for attrition prediction and employee risk analysis.

Columns:
Department, JobRole, MaritalStatus, OverTime, JobSatisfaction, Age, Attrition

Dataset:
[https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

---

### Promotion

Used for promotion eligibility and readiness scoring.

Columns:
employee_id, department, region, education, gender, recruitment_channel, no_of_trainings, age, previous_year_rating, length_of_service, awards_won, avg_training_score, is_promoted

Dataset:
[https://www.kaggle.com/datasets/arashnic/hr-ana](https://www.kaggle.com/datasets/arashnic/hr-ana)

---

## Resume screener

The resume screener helps rank candidates based on their resume content.

How to use:

* Upload resumes as `.txt` files
* Paste the job description
* Get ranked candidate scores

Example resume format:

```
Jane Smith
7 years experience in data engineering
Skills: Python, SQL, AWS, Kubernetes, leadership
Education: Master’s in Computer Science
Worked at Shopify and Accenture
```

---

## Key features

| Feature              | Module      | What it does                                             |
| -------------------- | ----------- | -------------------------------------------------------- |
| Culture matching     | Recruitment | Matches candidates with top-performing employee patterns |
| Resume ranking       | Recruitment | Scores resumes based on job description fit              |
| Attrition prediction | Retention   | Predicts employees likely to leave                       |
| Risk dashboard       | Retention   | Highlights high-risk employees for managers              |
| Promotion scoring    | Promotion   | Predicts promotion readiness and success likelihood      |
| Readiness timeline   | Promotion   | Shows when an employee may be promotion-ready            |
| HR analytics         | All modules | Overall hiring, retention, and promotion insights        |
| PDF reports          | Retention   | Downloadable employee risk reports                       |

---

## Tech stack

* Streamlit (frontend + backend UI)
* scikit-learn (ML models)
* pandas, numpy (data processing)
* matplotlib (visualization)
* seaborn (analysis charts)
* fpdf2 (PDF report generation)
* Streamlit Cloud (deployment)

---

## Model logic

Each module trains multiple machine learning models and automatically selects the best one based on accuracy.

The selected model is stored in session memory and used for predictions during runtime.

---

## Common issues

### Missing columns error

Make sure your dataset includes all required columns with correct spelling.

### Encoding error

Ensure dropdown values match dataset categories exactly.

### Low resume score

Clean your resume text before uploading (remove formatting issues from PDFs).

### App not running

Check that your virtual environment is active and dependencies are installed.



## Contributing

Feel free to fork the repository, raise issues, or submit pull requests. Contributions are always welcome.


