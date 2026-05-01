import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import re
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_curve, auc)

# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_resume_text(text: str) -> dict:
    """Extract simple signals from raw resume text."""
    text_lower = text.lower()

    tech_skills = ["python","java","sql","excel","tableau","power bi","machine learning",
                   "deep learning","aws","azure","react","node","tensorflow","pytorch",
                   "data analysis","statistics","r programming","c++","kubernetes","docker"]
    soft_skills = ["leadership","communication","teamwork","problem solving","management",
                   "collaboration","analytical","strategic","mentoring","negotiation"]

    found_tech  = [s for s in tech_skills  if s in text_lower]
    found_soft  = [s for s in soft_skills  if s in text_lower]

    edu_score = 0
    if any(w in text_lower for w in ["phd","doctorate"]):        edu_score = 4
    elif any(w in text_lower for w in ["master","mba","msc"]):   edu_score = 3
    elif any(w in text_lower for w in ["bachelor","bsc","b.sc","beng"]): edu_score = 2
    elif any(w in text_lower for w in ["diploma","associate"]):  edu_score = 1

    years = re.findall(r'(\d+)\s*\+?\s*year', text_lower)
    exp_years = max((int(y) for y in years), default=0)

    companies = len(re.findall(
        r'\b(inc\.|ltd\.|llc|corp\.|company|technologies|solutions|systems|group)\b',
        text_lower))

    return {
        "tech_skills":   found_tech,
        "soft_skills":   found_soft,
        "tech_score":    min(len(found_tech) * 10, 100),
        "soft_score":    min(len(found_soft) * 12, 100),
        "edu_score":     edu_score,
        "exp_years":     min(exp_years, 20),
        "companies":     companies,
    }


def compute_jd_match(resume: dict, jd_keywords: list[str]) -> float:
    all_skills = [s.lower() for s in resume["tech_skills"] + resume["soft_skills"]]
    if not jd_keywords:
        return 50.0
    matched = sum(1 for kw in jd_keywords if any(kw.lower() in s for s in all_skills))
    return round(matched / len(jd_keywords) * 100, 1)


def culture_dna_score(resume: dict, top_performer_profile: dict) -> float:
    """Compare candidate to top-performer fingerprint (0-100)."""
    score = 0
    score += min(resume["tech_score"],  top_performer_profile.get("avg_tech",  60)) / 100 * 35
    score += min(resume["soft_score"],  top_performer_profile.get("avg_soft",  50)) / 100 * 25
    score += min(resume["exp_years"],   top_performer_profile.get("avg_exp",    5)) / 20  * 25
    score += min(resume["edu_score"],   top_performer_profile.get("avg_edu",    2)) / 4   * 15
    return round(score * 100, 1)


def risk_badge(score: float) -> str:
    if score >= 70:   return "<span class='badge-low'>Strong Match</span>"
    if score >= 45:   return "<span class='badge-medium'>Moderate Match</span>"
    return "<span class='badge-high'>Weak Match</span>"


# ── Main ─────────────────────────────────────────────────────────────────────

def show():
    st.markdown("""
    <div class='page-header'>
        <h1>📄 Recruitment & Resume AI</h1>
        <p>Upload resumes, match against job descriptions, and rank candidates using Culture DNA fingerprinting of your top performers.</p>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["1 · Dataset & Train", "2 · Resume Screener", "3 · Department Hiring", "4 · EDA", "5 · Model Evaluation"])

    # ── TAB 1 — Dataset upload & model training ──────────────────────────
    with tabs[0]:
        st.markdown("### Upload Recruitment Dataset")
        st.markdown("""
        <div class='card'>
        <b>Required columns:</b> Age, Gender, EducationLevel, ExperienceYears, PreviousCompanies,
        DistanceFromCompany, InterviewScore, SkillScore, RecruitmentStrategy, HiringDecision, PersonalityScore
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="rec_csv")

        if uploaded:
            df = pd.read_csv(uploaded)
            df.dropna(inplace=True)

            required = ["Age","Gender","EducationLevel","ExperienceYears","PreviousCompanies",
                        "DistanceFromCompany","InterviewScore","SkillScore",
                        "RecruitmentStrategy","HiringDecision","PersonalityScore"]
            missing = [c for c in required if c not in df.columns]

            if missing:
                st.error(f"Missing columns: {missing}")
                st.stop()

            st.session_state["rec_data"] = df
            st.success(f"Dataset loaded — {len(df):,} rows")
            st.dataframe(df.head(8))

            # Build top-performer Culture DNA profile
            top = df[df["HiringDecision"] == 1]
            st.session_state["culture_dna"] = {
                "avg_tech": float(top["SkillScore"].mean()),
                "avg_soft": float(top["InterviewScore"].mean()),
                "avg_exp":  float(top["ExperienceYears"].mean()),
                "avg_edu":  float(top["EducationLevel"].mean()) if pd.api.types.is_numeric_dtype(top["EducationLevel"]) else 2.0,
            }
            st.info("✅ Culture DNA fingerprint built from your top performers.")

            # Prepare features
            X = df.drop(["HiringDecision","PersonalityScore"], axis=1)
            y = df["HiringDecision"]
            cat_cols = X.select_dtypes(include="object").columns
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

            le = LabelEncoder()
            y  = le.fit_transform(y)

            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

            st.session_state["rec_X"]      = X
            st.session_state["rec_Xtr"]    = X_tr
            st.session_state["rec_Xte"]    = X_te
            st.session_state["rec_ytr"]    = y_tr
            st.session_state["rec_yte"]    = y_te
            st.session_state["rec_scaler"] = sc
            st.session_state["rec_le"]     = le
            st.session_state["rec_cols"]   = list(X.columns)

            if st.button("🚀 Train Model"):
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_tr, y_tr)
                st.session_state["rec_model"] = model
                acc = accuracy_score(y_te, model.predict(X_te))
                st.success(f"Model trained! Accuracy: **{acc*100:.1f}%**")

    # ── TAB 2 — Resume Screener ──────────────────────────────────────────
    with tabs[1]:
        st.markdown("### AI Resume Screener")
        st.markdown("""
        <div class='card'>
        Upload one or multiple resumes (PDF or TXT). The AI reads them, extracts skills and
        experience, matches against your job description, and scores each candidate against
        your Culture DNA fingerprint.
        </div>
        """, unsafe_allow_html=True)

        jd_text = st.text_area(
            "Paste Job Description here",
            height=140,
            placeholder="e.g. We are looking for a Data Analyst with Python, SQL, Tableau, strong communication skills and 3+ years experience..."
        )

        jd_keywords = [w.strip() for w in jd_text.split(",") if len(w.strip()) > 2] if jd_text else []

        resume_files = st.file_uploader(
            "Upload Resumes (TXT files — one per candidate)",
            type=["txt"],
            accept_multiple_files=True,
            key="resumes"
        )

        if resume_files:
            dna = st.session_state.get("culture_dna", {
                "avg_tech": 60, "avg_soft": 50, "avg_exp": 5, "avg_edu": 2
            })

            results = []
            for f in resume_files:
                text = f.read().decode("utf-8", errors="ignore")
                parsed = parse_resume_text(text)
                jd_match  = compute_jd_match(parsed, jd_keywords)
                dna_score = culture_dna_score(parsed, dna)
                overall   = round((jd_match * 0.5 + dna_score * 0.5), 1)

                results.append({
                    "Candidate":       f.name.replace(".txt",""),
                    "JD Match (%)":    jd_match,
                    "Culture DNA (%)": dna_score,
                    "Overall Score":   overall,
                    "Tech Skills":     ", ".join(parsed["tech_skills"][:5]) or "—",
                    "Soft Skills":     ", ".join(parsed["soft_skills"][:3]) or "—",
                    "Exp Years":       parsed["exp_years"],
                    "Education Level": parsed["edu_score"],
                })

            results_df = pd.DataFrame(results).sort_values("Overall Score", ascending=False).reset_index(drop=True)
            results_df.index += 1

            st.markdown("#### Ranked Candidates")

            for _, row in results_df.iterrows():
                badge = risk_badge(row["Overall Score"])
                st.markdown(f"""
                <div class='card' style='margin-bottom:10px;'>
                    <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;'>
                        <div>
                            <div style='font-family:Syne,sans-serif;font-weight:700;color:#F1F5F9;font-size:1rem;'>
                                {row['Candidate']}
                            </div>
                            <div style='color:#64748B;font-size:0.82rem;margin-top:4px;'>
                                🛠 {row['Tech Skills']} &nbsp;|&nbsp; 🤝 {row['Soft Skills']}
                            </div>
                            <div style='color:#64748B;font-size:0.82rem;margin-top:2px;'>
                                📅 {row['Exp Years']} yrs exp &nbsp;|&nbsp; 🎓 Edu level {row['Education Level']}
                            </div>
                        </div>
                        <div style='text-align:right;'>
                            {badge}
                            <div style='margin-top:10px;display:flex;gap:16px;'>
                                <div style='text-align:center;'>
                                    <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;color:#38BDF8;'>{row['JD Match (%)']}%</div>
                                    <div style='color:#475569;font-size:0.7rem;'>JD Match</div>
                                </div>
                                <div style='text-align:center;'>
                                    <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;color:#A5B4FC;'>{row['Culture DNA (%)']}%</div>
                                    <div style='color:#475569;font-size:0.7rem;'>Culture DNA</div>
                                </div>
                                <div style='text-align:center;'>
                                    <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;color:#26DE81;'>{row['Overall Score']}%</div>
                                    <div style='color:#475569;font-size:0.7rem;'>Overall</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Download ranked list
            csv_bytes = results_df.to_csv(index=False).encode()
            st.download_button("⬇ Download Ranked Candidates CSV", csv_bytes,
                               "ranked_candidates.csv", "text/csv")

        else:
            st.info("Upload TXT resume files above to begin AI screening.")

    # ── TAB 3 — Department Hiring ────────────────────────────────────────
    with tabs[2]:
        st.markdown("### Department Hiring Selector")

        if "rec_data" not in st.session_state:
            st.warning("Please upload a dataset in Tab 1 first.")
            st.stop()

        df = st.session_state["rec_data"]

        strategy_map = {
            "Technical Hiring (Engineering / IT)": 1,
            "Corporate Hiring (HR / Admin)":       2,
            "Sales & Marketing Hiring":            3,
        }

        c1, c2 = st.columns(2)
        with c1:
            dept     = st.selectbox("Department", list(strategy_map.keys()))
            min_exp  = st.slider("Min Experience (years)", 0, int(df["ExperienceYears"].max()), 2)
        with c2:
            min_int  = st.slider("Min Interview Score", 0, 100, 40)
            min_skl  = st.slider("Min Skill Score",     0, 100, 50)

        strat_val = strategy_map[dept]
        dept_df   = df[
            (df["RecruitmentStrategy"] == strat_val) &
            (df["ExperienceYears"]     >= min_exp)   &
            (df["InterviewScore"]      >= min_int)   &
            (df["SkillScore"]          >= min_skl)
        ]

        st.markdown(f"**{len(dept_df)} eligible candidates** found")

        hire_n = st.slider("How many to hire?", 1, max(1, len(dept_df)),
                           min(3, len(dept_df)) if len(dept_df) >= 3 else 1)

        if st.button("🎯 Select Best Candidates") and "rec_model" in st.session_state:
            model  = st.session_state["rec_model"]
            scaler = st.session_state["rec_scaler"]
            cols   = st.session_state["rec_cols"]

            X_all = dept_df.drop(["HiringDecision","PersonalityScore"], axis=1, errors="ignore")
            X_all = pd.get_dummies(X_all)
            for col in cols:
                if col not in X_all.columns:
                    X_all[col] = 0
            X_all = scaler.transform(X_all[cols])

            probs = model.predict_proba(X_all)[:, 1]
            dept_df = dept_df.copy()
            dept_df["Hiring Score"] = (probs * 100).round(1)
            best = dept_df.sort_values("Hiring Score", ascending=False).head(hire_n)

            st.markdown("#### Top Selected Candidates")
            display_cols = [c for c in ["Age","Gender","EducationLevel","ExperienceYears",
                                        "InterviewScore","SkillScore","Hiring Score"]
                            if c in best.columns]
            st.dataframe(best[display_cols].reset_index(drop=True))
        elif "rec_model" not in st.session_state:
            st.info("Train the model in Tab 1 first to use this feature.")

    # ── TAB 4 — EDA ──────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("### Exploratory Data Analysis")

        if "rec_data" not in st.session_state:
            st.warning("Upload dataset in Tab 1 first.")
            st.stop()

        df = st.session_state["rec_data"]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.patch.set_facecolor("#0B0F1A")

        for ax in axes:
            ax.set_facecolor("#111827")
            ax.tick_params(colors="#94A3B8")
            for spine in ax.spines.values():
                spine.set_edgecolor("#1E2A40")

        # Hiring rate
        counts = df["HiringDecision"].value_counts(normalize=True) * 100
        axes[0].bar(["Not Hired","Hired"], counts.values, color=["#FF4757","#26DE81"])
        axes[0].set_title("Hiring Rate (%)", color="#F1F5F9", fontsize=11)

        # Experience distribution
        axes[1].hist(df["ExperienceYears"], bins=15, color="#38BDF8", edgecolor="#0B0F1A")
        axes[1].set_title("Experience Distribution", color="#F1F5F9", fontsize=11)

        # Skill vs Interview score scatter
        hired    = df[df["HiringDecision"] == 1]
        not_hired = df[df["HiringDecision"] == 0]
        axes[2].scatter(not_hired["SkillScore"], not_hired["InterviewScore"],
                        alpha=0.4, s=20, c="#FF4757", label="Not Hired")
        axes[2].scatter(hired["SkillScore"],     hired["InterviewScore"],
                        alpha=0.6, s=20, c="#26DE81", label="Hired")
        axes[2].set_xlabel("Skill Score", color="#94A3B8")
        axes[2].set_ylabel("Interview Score", color="#94A3B8")
        axes[2].set_title("Skill vs Interview (by outcome)", color="#F1F5F9", fontsize=11)
        axes[2].legend(facecolor="#111827", labelcolor="#94A3B8")

        plt.tight_layout()
        st.pyplot(fig)

    # ── TAB 5 — Model Evaluation ─────────────────────────────────────────
    with tabs[4]:
        st.markdown("### Model Comparison & ROC Curves")

        if "rec_Xtr" not in st.session_state:
            st.warning("Upload and process dataset in Tab 1 first.")
            st.stop()

        X_tr = st.session_state["rec_Xtr"]
        X_te = st.session_state["rec_Xte"]
        y_tr = st.session_state["rec_ytr"]
        y_te = st.session_state["rec_yte"]

        models_eval = {
            "Logistic Regression":  LogisticRegression(max_iter=200),
            "Random Forest":        RandomForestClassifier(),
            "Decision Tree":        DecisionTreeClassifier(),
            "Gradient Boosting":    GradientBoostingClassifier(),
        }

        results = []
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#0B0F1A")
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#94A3B8")
        for spine in ax.spines.values(): spine.set_edgecolor("#1E2A40")

        colors = ["#38BDF8","#A5B4FC","#26DE81","#FF9F43"]

        for (name, model), color in zip(models_eval.items(), colors):
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1]

            fpr, tpr, _ = roc_curve(y_te, y_prob)
            roc_auc     = auc(fpr, tpr)

            ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})", color=color, lw=2)

            results.append({
                "Model":     name,
                "Accuracy":  round(accuracy_score(y_te, y_pred)  * 100, 2),
                "Precision": round(precision_score(y_te, y_pred) * 100, 2),
                "Recall":    round(recall_score(y_te, y_pred)    * 100, 2),
                "F1 Score":  round(f1_score(y_te, y_pred)        * 100, 2),
                "AUC":       round(roc_auc * 100, 2),
            })

        ax.plot([0,1],[0,1],"--", color="#334155")
        ax.set_xlabel("False Positive Rate", color="#94A3B8")
        ax.set_ylabel("True Positive Rate",  color="#94A3B8")
        ax.set_title("ROC Curve Comparison",  color="#F1F5F9")
        ax.legend(facecolor="#111827", labelcolor="#94A3B8")

        st.pyplot(fig)
        st.dataframe(pd.DataFrame(results).set_index("Model"))
