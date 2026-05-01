# HR Navigator — Workforce Intelligence Platform

An end-to-end AI-powered HR analytics platform covering three core domains: recruitment prediction, attrition risk analysis, and promotion prediction. Built on Streamlit with scikit-learn ML models, matplotlib visualisations, and a PDF export layer.

---

## Project structure

```
hr-navigator/
│
├── app.py                      Main entry point — run this file
│
├── _pages/
│   ├── __init__.py
│   ├── home.py                 Landing page and employee journey overview
│   ├── recruitment.py          Recruitment dataset training, resume AI screener, department hiring
│   ├── retention.py            Attrition prediction, life event signals, team dashboard
│   ├── promotion.py            Promotion eligibility, regret score, department allocation
│   └── analytics.py            Cross-module platform analytics
│
├── .streamlit/
│   └── config.toml             Theme and server configuration
│
├── requirements.txt            Python dependencies
├── .gitignore
└── README.md
```

---

## How to run locally

**Step 1 — Confirm Python version**

Python 3.10 or higher is required.

```bash
python --version
```

Download from https://www.python.org/downloads/ if needed.

**Step 2 — Get the project**

Clone via Git:

```bash
git clone https://github.com/YOUR_USERNAME/hr-navigator.git
cd hr-navigator
```

Or download the ZIP from GitHub and extract it, then open a terminal inside the folder.

**Step 3 — Create a virtual environment**

```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Mac / Linux:
source venv/bin/activate
```

**Step 4 — Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 5 — Launch the app**

```bash
streamlit run app.py
```

The app opens automatically at http://localhost:8501

---

## Deploy to Streamlit Cloud (free)

Streamlit Cloud hosts your app publicly at no cost and redeploys automatically on every push to GitHub.

**Step 1 — Push to GitHub**

Create a repository at https://github.com/new. Name it `hr-navigator`, set it to Public, then push:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/hr-navigator.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

**Step 2 — Connect to Streamlit Cloud**

1. Go to https://share.streamlit.io and sign in with GitHub
2. Click **New app**
3. Set the fields:
   - Repository: `YOUR_USERNAME/hr-navigator`
   - Branch: `main`
   - Main file path: `app.py`
4. Click **Deploy**

Deployment takes roughly two to three minutes. Your app will be live at a public URL you can share.

---

## Datasets

Each module expects a specific CSV structure. Sample datasets are linked below.

### Recruitment module

Required columns:

```
Age, Gender, EducationLevel, ExperienceYears, PreviousCompanies,
DistanceFromCompany, InterviewScore, SkillScore, RecruitmentStrategy,
HiringDecision (0 or 1), PersonalityScore
```

Sample dataset: https://www.kaggle.com/datasets/rabieelkharoua/predicting-hiring-decisions-in-recruitment-data

### Retention module

Required columns:

```
Department, JobRole, MaritalStatus, OverTime, JobSatisfaction,
Age, Attrition (Yes / No)
```

Sample dataset: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

### Promotion module

Required columns:

```
employee_id, department, region, education, gender,
recruitment_channel, no_of_trainings, age, previous_year_rating,
length_of_service, awards_won, avg_training_score, is_promoted (0 or 1)
```

Sample dataset: https://www.kaggle.com/datasets/arashnic/hr-ana

---

## Resume screener

The resume screener accepts plain text files (.txt), one file per candidate.

To use it:

1. Save each resume as a .txt file
2. Upload all files in the Resume Screener tab
3. Paste the job description into the text box
4. The system scores and ranks candidates automatically

Example resume format:

```
Jane Smith
7 years experience in data engineering
Skills: Python, SQL, AWS, Kubernetes, leadership, communication
Education: Master of Science in Computer Science
Previous employers: Shopify, Accenture
```

---

## Key features

| Feature | Module | Description |
|---|---|---|
| Culture DNA matching | Recruitment | Matches candidates against a fingerprint built from your top performers, not just a job description |
| Resume AI screener | Recruitment | Parses plain-text resumes, extracts skills and experience, ranks candidates by JD fit and culture fit |
| Life event signals | Retention | Applies personal life event multipliers (relocation, new baby, degree completion) to base attrition scores |
| Early warning dashboard | Retention | Team-level risk overview with colour-coded risk levels and manager-specific recommended actions |
| Promotion regret score | Promotion | Projects 18-month post-promotion success, not just current eligibility |
| Readiness timeline | Promotion | States when an employee will be ready for promotion, not just whether they qualify |
| Platform analytics | All modules | Cross-module summary: hire rate, attrition rate, promotion rate, and full lifecycle connectivity status |
| PDF export | Retention | Generates a downloadable attrition risk report for individual employees |

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend and app server | Streamlit |
| Machine learning | scikit-learn (Random Forest, Gradient Boosting, Logistic Regression, Decision Tree, SVM) |
| Data processing | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| PDF export | fpdf2 |
| Deployment | Streamlit Cloud |

---

## Model selection logic

Each module trains multiple candidate models on your uploaded dataset and automatically selects the one with the highest accuracy on the held-out test set (20% split). The selected model is stored in session state and used for all subsequent predictions in that session.

---

## Common issues

**Missing column error on upload** — confirm your CSV has all required columns listed above. Column names are case-sensitive.

**Encoding error in the promotion predictor** — ensure that the values you select in the dropdowns match values that appear in your uploaded dataset. The encoder cannot handle unseen categories.

**Resume screener shows low scores** — the screener parses plain text. If your files contain formatting artifacts from PDF-to-text conversion, clean them before uploading.

**App not starting locally** — confirm the virtual environment is active and all packages installed without errors. Run `pip install -r requirements.txt` again if needed.

---

## Contributing

Open an issue or pull request on GitHub.
