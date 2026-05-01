import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_curve, auc)


# ── Helpers ──────────────────────────────────────────────────────────────────

READINESS_SIGNALS = {
    "Has a senior mentor in the company": 0.12,
    "Led a cross-functional project":     0.10,
    "Completed management training":      0.09,
    "Received peer recognition award":    0.08,
    "Took on extra responsibilities voluntarily": 0.07,
}

RISK_FLAGS = {
    "No mentor relationship":             -0.12,
    "Promoted less than 18 months ago":   -0.15,
    "Team has high attrition currently":  -0.10,
    "Performance dipped last quarter":    -0.13,
    "No peer support network identified": -0.08,
}

def readiness_timeline(score: float) -> str:
    if score >= 80: return "Ready Now"
    if score >= 60: return "Ready in ~6 months"
    if score >= 40: return "Ready in ~12 months"
    return "Needs development plan (12–24 months)"


def post_promotion_risk(base_prob: float, signals: list, risks: list) -> float:
    adjustment = sum(READINESS_SIGNALS[s] for s in signals if s in READINESS_SIGNALS)
    adjustment += sum(RISK_FLAGS[r] for r in risks if r in RISK_FLAGS)
    return round(min(max(base_prob + adjustment, 0.01), 0.99) * 100, 1)


def prepare_promotion_data(df: pd.DataFrame):
    df = df.dropna()
    features = [
        "employee_id","department","region","education","gender",
        "recruitment_channel","no_of_trainings","age",
        "previous_year_rating","length_of_service","awards_won","avg_training_score"
    ]
    features = [f for f in features if f in df.columns]

    cat_cols = df[features].select_dtypes(include="object").columns.tolist()
    encoder  = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough"
    )

    X  = encoder.fit_transform(df[features])
    le = LabelEncoder()
    y  = le.fit_transform(df["is_promoted"])

    return X, y, encoder, features, df


# ── Main ─────────────────────────────────────────────────────────────────────

def show():
    st.markdown("""
    <div class='page-header'>
        <h1>🏆 Promotion Intelligence</h1>
        <p>Predict post-promotion success, not just eligibility. Identify who will thrive 18 months after promotion — and who you will regret promoting.</p>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["1 · Upload & Train", "2 · Promotion Predictor", "3 · Regret Score", "4 · Department Allocation", "5 · EDA & Evaluation"])

    # ── TAB 1 — Upload ────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("### Upload Employee Dataset")
        st.markdown("""
        <div class='card'>
        <b>Required columns:</b> employee_id, department, region, education, gender,
        recruitment_channel, no_of_trainings, age, previous_year_rating,
        length_of_service, awards_won, avg_training_score, is_promoted (0/1)
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="prom_csv")

        if uploaded:
            raw_df = pd.read_csv(uploaded)

            if "is_promoted" not in raw_df.columns:
                st.error("Dataset must contain 'is_promoted' column (0 or 1).")
                st.stop()

            X, y, encoder, features, df = prepare_promotion_data(raw_df)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

            st.session_state.update({
                "prom_df":    df,
                "prom_X":     X,
                "prom_y":     y,
                "prom_Xtr":   X_tr,
                "prom_Xte":   X_te,
                "prom_ytr":   y_tr,
                "prom_yte":   y_te,
                "prom_enc":   encoder,
                "prom_feats": features,
            })

            st.success(f"Dataset loaded — {len(df):,} rows")
            st.dataframe(df.head(8))

            if st.button("🚀 Train Best Model"):
                candidates = {
                    "Random Forest":     RandomForestClassifier(n_estimators=100, random_state=42),
                    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                    "Logistic Regression": LogisticRegression(max_iter=200),
                }
                best_name, best_model, best_acc = "", None, 0
                for name, m in candidates.items():
                    m.fit(X_tr, y_tr)
                    acc = accuracy_score(y_te, m.predict(X_te))
                    if acc > best_acc:
                        best_acc, best_model, best_name = acc, m, name

                st.session_state["prom_model"]      = best_model
                st.session_state["prom_model_name"] = best_name
                st.success(f"Best model: **{best_name}** — Accuracy: **{best_acc*100:.1f}%**")

    # ── TAB 2 — Promotion Predictor ───────────────────────────────────────
    with tabs[1]:
        st.markdown("### Individual Promotion Eligibility")

        if "prom_df" not in st.session_state:
            st.warning("Upload dataset in Tab 1 first.")
            st.stop()

        df      = st.session_state["prom_df"]
        encoder = st.session_state["prom_enc"]
        feats   = st.session_state["prom_feats"]

        c1, c2 = st.columns(2)
        with c1:
            dept     = st.selectbox("Department",  df["department"].unique()  if "department"  in df.columns else ["N/A"])
            edu      = st.selectbox("Education",   df["education"].unique()   if "education"   in df.columns else ["N/A"])
            channel  = st.selectbox("Recruitment Channel", df["recruitment_channel"].unique() if "recruitment_channel" in df.columns else ["N/A"])
        with c2:
            trainings = st.slider("No. of Trainings Completed", 0, 10, 2)
            rating    = st.slider("Previous Year Rating (1–5)", 1, 5, 3)
            service   = st.slider("Length of Service (years)",  1, 30, 5)
            awards    = st.number_input("Awards Won", 0, 10, 0)
            train_score = st.slider("Avg Training Score", 40, 100, 65)
            age       = st.slider("Age", 20, 65, 32)

        if st.button("📊 Predict Promotion Eligibility") and "prom_model" in st.session_state:
            model = st.session_state["prom_model"]

            # Build input row matching feature list
            row_dict = {
                "employee_id": 0, "department": dept, "region": "region_1",
                "education": edu, "gender": "m", "recruitment_channel": channel,
                "no_of_trainings": trainings, "age": age,
                "previous_year_rating": rating, "length_of_service": service,
                "awards_won": awards, "avg_training_score": train_score,
            }
            input_df = pd.DataFrame([{f: row_dict.get(f, 0) for f in feats}])
            try:
                input_enc = encoder.transform(input_df)
            except Exception as e:
                st.error(f"Encoding error: {e}")
                st.stop()

            prob      = model.predict_proba(input_enc)[0][1]
            score     = round(prob * 100, 1)
            timeline  = readiness_timeline(score)
            color     = "#26DE81" if score >= 70 else ("#FF9F43" if score >= 45 else "#FF4757")

            st.markdown(f"""
            <div class='card' style='border-color:{color};'>
                <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;'>
                    <div>
                        <div style='font-family:Syne,sans-serif;font-size:2.4rem;font-weight:800;color:{color};'>
                            {score}%
                        </div>
                        <div style='color:#94A3B8;font-size:0.9rem;'>Promotion Readiness Score</div>
                        <div style='margin-top:12px;font-size:0.95rem;color:#F1F5F9;'>
                            🗓 <b>Timeline:</b> {timeline}
                        </div>
                    </div>
                    <div style='background:#0F1628;border-radius:12px;padding:20px;min-width:220px;'>
                        <div style='color:#64748B;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>
                            Key Inputs
                        </div>
                        <div style='color:#94A3B8;font-size:0.85rem;line-height:1.8;'>
                            🏢 {dept}<br>
                            📅 {service} yrs service<br>
                            ⭐ Rating: {rating}/5<br>
                            🏋 {trainings} trainings<br>
                            🏆 {awards} award(s)
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif "prom_model" not in st.session_state:
            st.info("Train the model in Tab 1 first.")

    # ── TAB 3 — Promotion Regret Score ───────────────────────────────────
    with tabs[2]:
        st.markdown("### Promotion Regret Score")
        st.markdown("""
        <div class='card'>
        <b>The unique feature nobody else has.</b> Before promoting someone, predict whether
        they will still be succeeding 18 months from now — or whether this will be a
        promotion you regret.
        </div>
        """, unsafe_allow_html=True)

        base_score = st.slider("Employee's Promotion Readiness Score (%)", 0, 100, 72)

        st.markdown("#### ✅ Positive Readiness Signals")
        pos_signals = st.multiselect("Select signals that apply:", list(READINESS_SIGNALS.keys()))

        st.markdown("#### ⚠️ Risk Flags")
        neg_flags = st.multiselect("Select risk flags that apply:", list(RISK_FLAGS.keys()))

        if st.button("🔮 Calculate Post-Promotion Success Score"):
            base_prob     = base_score / 100
            final_score   = post_promotion_risk(base_prob, pos_signals, neg_flags)
            delta         = round(final_score - base_score, 1)
            delta_color   = "#26DE81" if delta >= 0 else "#FF4757"
            delta_label   = f"+{delta}%" if delta >= 0 else f"{delta}%"

            color = "#26DE81" if final_score >= 70 else ("#FF9F43" if final_score >= 45 else "#FF4757")

            verdict = "✅ Promote with confidence" if final_score >= 70 \
                 else ("⚠️ Proceed with support plan" if final_score >= 50 \
                 else "❌ Hold — address risk flags first")

            st.markdown(f"""
            <div class='card' style='border-color:{color};'>
                <div style='display:flex;justify-content:space-around;align-items:center;flex-wrap:wrap;gap:16px;text-align:center;'>
                    <div>
                        <div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#94A3B8;'>
                            {base_score}%
                        </div>
                        <div style='color:#475569;font-size:0.8rem;'>Eligibility Score</div>
                    </div>
                    <div style='color:#334155;font-size:1.8rem;'>→</div>
                    <div>
                        <div style='font-family:Syne,sans-serif;font-size:2.4rem;font-weight:800;color:{color};'>
                            {final_score}%
                        </div>
                        <div style='color:#475569;font-size:0.8rem;'>18-Month Success Score</div>
                        <div style='color:{delta_color};font-size:0.85rem;font-weight:600;margin-top:4px;'>{delta_label}</div>
                    </div>
                    <div style='background:#0F1628;border-radius:12px;padding:16px 24px;'>
                        <div style='color:#F1F5F9;font-size:1rem;font-weight:600;'>{verdict}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if neg_flags:
                st.markdown("#### 📋 Recommended Actions Before Promoting")
                action_map = {
                    "No mentor relationship":             "Assign a senior mentor for 90 days before promotion.",
                    "Promoted less than 18 months ago":   "Wait at least 6 more months for role stabilization.",
                    "Team has high attrition currently":  "Stabilize team dynamics before adding role change pressure.",
                    "Performance dipped last quarter":    "Investigate root cause — ensure this is temporary.",
                    "No peer support network identified": "Facilitate introductions across departments first.",
                }
                for flag in neg_flags:
                    if flag in action_map:
                        st.markdown(f"- **{flag}:** {action_map[flag]}")

    # ── TAB 4 — Department Allocation ────────────────────────────────────
    with tabs[3]:
        st.markdown("### Department Promotion Allocation")

        if "prom_df" not in st.session_state or "prom_model" not in st.session_state:
            st.warning("Upload dataset and train model in Tab 1 first.")
            st.stop()

        df      = st.session_state["prom_df"]
        model   = st.session_state["prom_model"]
        encoder = st.session_state["prom_enc"]
        feats   = st.session_state["prom_feats"]

        dept_choice = st.selectbox("Select Department",
                                   df["department"].unique() if "department" in df.columns else ["N/A"])
        seats       = st.number_input("Promotion Seats Available", 1, 20, 3)

        if st.button("🏆 Identify Top Promotion Candidates"):
            dept_df = df[df["department"] == dept_choice].copy()

            try:
                X_dept = encoder.transform(dept_df[feats])
                probs  = model.predict_proba(X_dept)[:, 1]
                dept_df["Promotion Score (%)"] = (probs * 100).round(1)

                top_candidates = dept_df.sort_values(
                    "Promotion Score (%)", ascending=False
                ).head(seats)

                show_cols = [c for c in ["employee_id","department","age","length_of_service",
                                         "previous_year_rating","avg_training_score",
                                         "awards_won","Promotion Score (%)"]
                             if c in top_candidates.columns]

                st.markdown(f"#### Top {seats} Candidates in {dept_choice}")
                st.dataframe(top_candidates[show_cols].reset_index(drop=True))

                csv = top_candidates[show_cols].to_csv(index=False).encode()
                st.download_button("⬇ Download Candidates CSV", csv,
                                   "promotion_candidates.csv", "text/csv")

            except Exception as e:
                st.error(f"Scoring failed: {e}")

    # ── TAB 5 — EDA & Evaluation ──────────────────────────────────────────
    with tabs[4]:
        st.markdown("### EDA & Model Evaluation")

        if "prom_df" not in st.session_state:
            st.warning("Upload dataset in Tab 1 first.")
            st.stop()

        df = st.session_state["prom_df"]

        # EDA charts
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.patch.set_facecolor("#0B0F1A")
        for ax in axes:
            ax.set_facecolor("#111827")
            ax.tick_params(colors="#94A3B8", labelsize=8)
            for spine in ax.spines.values(): spine.set_edgecolor("#1E2A40")

        if "department" in df.columns and "is_promoted" in df.columns:
            prom_rate = df.groupby("department")["is_promoted"].mean()*100
            axes[0].barh(prom_rate.index, prom_rate.values, color="#38BDF8")
            axes[0].set_title("Promotion Rate by Dept (%)", color="#F1F5F9", fontsize=10)

        if "avg_training_score" in df.columns and "is_promoted" in df.columns:
            promoted     = df[df["is_promoted"]==1]["avg_training_score"]
            not_promoted = df[df["is_promoted"]==0]["avg_training_score"]
            axes[1].hist(not_promoted, bins=20, alpha=0.6, color="#FF4757", label="Not Promoted")
            axes[1].hist(promoted,     bins=20, alpha=0.7, color="#26DE81", label="Promoted")
            axes[1].set_title("Training Score Distribution", color="#F1F5F9", fontsize=10)
            axes[1].legend(facecolor="#111827", labelcolor="#94A3B8")

        if "length_of_service" in df.columns and "is_promoted" in df.columns:
            for val, color, label in [(1,"#26DE81","Promoted"),(0,"#FF4757","Not Promoted")]:
                axes[2].hist(df[df["is_promoted"]==val]["length_of_service"],
                             bins=15, alpha=0.7, color=color, label=label)
            axes[2].set_title("Length of Service vs Promotion", color="#F1F5F9", fontsize=10)
            axes[2].legend(facecolor="#111827", labelcolor="#94A3B8")

        plt.tight_layout()
        st.pyplot(fig)

        # Model evaluation
        if "prom_Xtr" in st.session_state:
            st.markdown("### Model Comparison")

            X_tr = st.session_state["prom_Xtr"]
            X_te = st.session_state["prom_Xte"]
            y_tr = st.session_state["prom_ytr"]
            y_te = st.session_state["prom_yte"]

            eval_models = {
                "Logistic Regression": LogisticRegression(max_iter=200),
                "Random Forest":       RandomForestClassifier(),
                "Decision Tree":       DecisionTreeClassifier(),
                "Gradient Boosting":   GradientBoostingClassifier(),
            }

            results = []
            fig2, ax2 = plt.subplots(figsize=(8,5))
            fig2.patch.set_facecolor("#0B0F1A")
            ax2.set_facecolor("#111827")
            ax2.tick_params(colors="#94A3B8")
            for spine in ax2.spines.values(): spine.set_edgecolor("#1E2A40")

            colors = ["#38BDF8","#A5B4FC","#26DE81","#FF9F43"]

            for (name, m), color in zip(eval_models.items(), colors):
                m.fit(X_tr, y_tr)
                y_pred = m.predict(X_te)
                y_prob = m.predict_proba(X_te)[:, 1]
                fpr, tpr, _ = roc_curve(y_te, y_prob)
                roc_auc     = auc(fpr, tpr)

                ax2.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})", color=color, lw=2)
                results.append({
                    "Model":     name,
                    "Accuracy":  round(accuracy_score(y_te, y_pred)  * 100, 2),
                    "Precision": round(precision_score(y_te, y_pred) * 100, 2),
                    "Recall":    round(recall_score(y_te, y_pred)    * 100, 2),
                    "F1 Score":  round(f1_score(y_te, y_pred)        * 100, 2),
                    "AUC":       round(roc_auc * 100, 2),
                })

            ax2.plot([0,1],[0,1],"--", color="#334155")
            ax2.set_xlabel("False Positive Rate", color="#94A3B8")
            ax2.set_ylabel("True Positive Rate",  color="#94A3B8")
            ax2.set_title("ROC Curve Comparison",  color="#F1F5F9")
            ax2.legend(facecolor="#111827", labelcolor="#94A3B8")

            st.pyplot(fig2)
            st.dataframe(pd.DataFrame(results).set_index("Model"))
