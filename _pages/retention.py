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
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_curve, auc, confusion_matrix)
from fpdf import FPDF
import io


# ── Helpers ──────────────────────────────────────────────────────────────────

LIFE_EVENT_MULTIPLIERS = {
    "None":                               1.00,
    "Recently had a baby / family change": 1.35,
    "Spouse relocated / partner job change":1.40,
    "Just completed a degree / certification":1.25,
    "Recently turned 30 or 40":           1.15,
    "Going through a divorce / separation":1.30,
    "Financial stress or housing change":  1.20,
}

RETENTION_ACTIONS = {
    "Recently had a baby / family change":     "Offer flexible / remote work arrangement",
    "Spouse relocated / partner job change":   "Explore relocation support or remote options",
    "Just completed a degree / certification": "Discuss career growth path and salary review",
    "Recently turned 30 or 40":               "Schedule career milestone conversation",
    "Going through a divorce / separation":    "Connect with EAP (Employee Assistance Program)",
    "Financial stress or housing change":      "Review compensation and benefits package",
    "None":                                    "Schedule a regular 1-on-1 stay interview",
}

def risk_level(score: float):
    if score > 70: return "HIGH",   "#FF4757"
    if score > 45: return "MEDIUM", "#FF9F43"
    return "LOW", "#26DE81"


def prepare_attrition_data(df: pd.DataFrame):
    drop_cols = [c for c in ["EmployeeNumber","StockOptionLevel"] if c in df.columns]
    df = df.drop(columns=drop_cols).dropna()

    features = ["Department","JobRole","MaritalStatus","OverTime",
                "JobSatisfaction","Age"]
    features = [f for f in features if f in df.columns]

    cat_cols = df[features].select_dtypes(include="object").columns.tolist()
    encoder  = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough"
    )

    X   = encoder.fit_transform(df[features])
    le  = LabelEncoder()
    y   = le.fit_transform(df["Attrition"])

    return X, y, encoder, features, df


# ── Main ─────────────────────────────────────────────────────────────────────

def show():
    st.markdown("""
    <div class='page-header'>
        <h1>🔄 Retention & Early Warning System</h1>
        <p>Monitor attrition risk continuously. Life event signals, manager impact scoring, and automated alerts with specific recommended actions.</p>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["1 · Upload & Train", "2 · Individual Risk Check", "3 · Team Dashboard", "4 · EDA", "5 · Model Evaluation"])

    # ── TAB 1 — Upload ────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("### Upload Employee Dataset")
        st.markdown("""
        <div class='card'>
        <b>Required columns:</b> Department, JobRole, MaritalStatus, OverTime,
        JobSatisfaction, Age, Attrition (Yes/No)
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="ret_csv")

        if uploaded:
            raw_df = pd.read_csv(uploaded)

            if "Attrition" not in raw_df.columns:
                st.error("Dataset must contain an 'Attrition' column (Yes/No).")
                st.stop()

            X, y, encoder, features, df = prepare_attrition_data(raw_df)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

            st.session_state.update({
                "ret_raw":     raw_df,
                "ret_df":      df,
                "ret_X":       X,
                "ret_y":       y,
                "ret_Xtr":     X_tr,
                "ret_Xte":     X_te,
                "ret_ytr":     y_tr,
                "ret_yte":     y_te,
                "ret_enc":     encoder,
                "ret_feats":   features,
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

                st.session_state["ret_model"]      = best_model
                st.session_state["ret_model_name"] = best_name
                st.success(f"Best model: **{best_name}** — Accuracy: **{best_acc*100:.1f}%**")

    # ── TAB 2 — Individual Risk Check ─────────────────────────────────────
    with tabs[1]:
        st.markdown("### Individual Employee Risk Check")

        if "ret_df" not in st.session_state:
            st.warning("Upload dataset in Tab 1 first.")
            st.stop()

        df      = st.session_state["ret_df"]
        encoder = st.session_state["ret_enc"]
        feats   = st.session_state["ret_feats"]

        c1, c2 = st.columns(2)
        with c1:
            dept   = st.selectbox("Department",    df["Department"].unique()    if "Department"   in df.columns else ["N/A"])
            role   = st.selectbox("Job Role",       df["JobRole"].unique()       if "JobRole"      in df.columns else ["N/A"])
            marital= st.selectbox("Marital Status", df["MaritalStatus"].unique() if "MaritalStatus"in df.columns else ["N/A"])
        with c2:
            overtime = st.selectbox("OverTime", df["OverTime"].unique() if "OverTime" in df.columns else ["Yes","No"])
            jobsat   = st.slider("Job Satisfaction (1–4)", 1, 4, 3)
            age      = st.slider("Employee Age", 18, 65, 32)

        st.markdown("---")
        st.markdown("#### 🌱 Life Event Check-in *(voluntary — improves prediction accuracy)*")
        life_event = st.selectbox("Any major life change this quarter?", list(LIFE_EVENT_MULTIPLIERS.keys()))

        if st.button("⚡ Predict Attrition Risk") and "ret_model" in st.session_state:
            model = st.session_state["ret_model"]

            input_df = pd.DataFrame([[dept, role, marital, overtime, jobsat, age]],
                                     columns=feats[:6] if len(feats) >= 6 else feats)
            try:
                input_enc = encoder.transform(input_df[feats])
            except Exception:
                st.error("Input encoding failed — check that your selections match the dataset values.")
                st.stop()

            base_prob    = model.predict_proba(input_enc)[0][1]
            multiplier   = LIFE_EVENT_MULTIPLIERS[life_event]
            adjusted_prob = min(base_prob * multiplier, 0.99)
            risk_pct      = round(adjusted_prob * 100, 1)
            level, color  = risk_level(risk_pct)
            action        = RETENTION_ACTIONS[life_event]

            st.markdown(f"""
            <div class='card' style='border-color:{color};'>
                <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;'>
                    <div>
                        <div style='font-family:Syne,sans-serif;font-size:2.4rem;font-weight:800;color:{color};'>
                            {risk_pct}%
                        </div>
                        <div style='color:#94A3B8;font-size:0.9rem;margin-top:4px;'>Attrition Risk Score</div>
                        <div style='margin-top:12px;font-size:0.95rem;color:#F1F5F9;'>
                            <b>Risk Level:</b> <span style='color:{color};font-weight:700;'>{level}</span>
                        </div>
                        <div style='margin-top:8px;font-size:0.9rem;color:#94A3B8;'>
                            <b>Life Event Impact:</b> {life_event}
                        </div>
                    </div>
                    <div style='background:#0F1628;border-radius:12px;padding:20px;min-width:260px;'>
                        <div style='color:#64748B;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>
                            Recommended Action
                        </div>
                        <div style='color:#F1F5F9;font-size:0.95rem;font-weight:500;'>
                            💡 {action}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if risk_pct > 70:
                st.error("⚠️ High Risk — Immediate manager conversation recommended within 2 weeks.")
            elif risk_pct > 45:
                st.warning("⚠️ Medium Risk — Schedule a stay interview within 30 days.")
            else:
                st.success("✅ Low Risk — Continue regular check-ins.")

            # PDF report
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial","B",16)
            pdf.cell(0, 10, "HR Attrition Risk Report", ln=True, align="C")
            pdf.set_font("Arial","",12)
            pdf.ln(8)
            pdf.cell(0, 8, f"Department:      {dept}",       ln=True)
            pdf.cell(0, 8, f"Job Role:        {role}",       ln=True)
            pdf.cell(0, 8, f"Age:             {age}",        ln=True)
            pdf.cell(0, 8, f"Marital Status:  {marital}",   ln=True)
            pdf.cell(0, 8, f"OverTime:        {overtime}",  ln=True)
            pdf.cell(0, 8, f"Job Satisfaction:{jobsat}",     ln=True)
            pdf.cell(0, 8, f"Life Event:      {life_event}", ln=True)
            pdf.ln(6)
            pdf.set_font("Arial","B",13)
            pdf.cell(0, 8, f"Attrition Risk Score: {risk_pct}% — {level}", ln=True)
            pdf.set_font("Arial","",12)
            pdf.ln(4)
            pdf.multi_cell(0, 8, f"Recommended Action: {action}")

            pdf_bytes = io.BytesIO(pdf.output(dest="S").encode("latin-1"))
            st.download_button("⬇ Download Risk Report PDF", pdf_bytes,
                               f"risk_report_{role}.pdf", "application/pdf")

        elif "ret_model" not in st.session_state:
            st.info("Train the model in Tab 1 first.")

    # ── TAB 3 — Team Dashboard ────────────────────────────────────────────
    with tabs[2]:
        st.markdown("### Team Risk Dashboard")

        if "ret_df" not in st.session_state or "ret_model" not in st.session_state:
            st.warning("Upload dataset and train model in Tab 1 first.")
            st.stop()

        df      = st.session_state["ret_df"]
        model   = st.session_state["ret_model"]
        encoder = st.session_state["ret_enc"]
        feats   = st.session_state["ret_feats"]

        dept_filter = st.selectbox("Filter by Department",
                                   ["All"] + list(df["Department"].unique()) if "Department" in df.columns else ["All"])

        view_df = df if dept_filter == "All" else df[df["Department"] == dept_filter]

        try:
            X_all  = encoder.transform(view_df[feats])
            probs  = model.predict_proba(X_all)[:, 1]
            view_df = view_df.copy()
            view_df["Risk Score (%)"] = (probs * 100).round(1)
            view_df["Risk Level"]     = view_df["Risk Score (%)"].apply(
                lambda s: "🔴 HIGH" if s > 70 else ("🟠 MEDIUM" if s > 45 else "🟢 LOW")
            )

            high_risk   = (view_df["Risk Score (%)"] > 70).sum()
            medium_risk = ((view_df["Risk Score (%)"] > 45) & (view_df["Risk Score (%)"] <= 70)).sum()
            low_risk    = (view_df["Risk Score (%)"] <= 45).sum()

            m1, m2, m3, m4 = st.columns(4)
            for col, val, label, color in [
                (m1, len(view_df),   "Total Employees", "#38BDF8"),
                (m2, high_risk,      "High Risk",       "#FF4757"),
                (m3, medium_risk,    "Medium Risk",     "#FF9F43"),
                (m4, low_risk,       "Low Risk",        "#26DE81"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class='metric-tile'>
                        <div class='metric-value' style='color:{color};'>{val}</div>
                        <div class='metric-label'>{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            display_cols = [c for c in ["Department","JobRole","Age","OverTime",
                                        "JobSatisfaction","Risk Score (%)","Risk Level"]
                            if c in view_df.columns]
            st.dataframe(
                view_df[display_cols].sort_values("Risk Score (%)", ascending=False).reset_index(drop=True),
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Could not score employees: {e}")

    # ── TAB 4 — EDA ──────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("### Exploratory Data Analysis")

        if "ret_df" not in st.session_state:
            st.warning("Upload dataset in Tab 1 first.")
            st.stop()

        df = st.session_state["ret_df"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.patch.set_facecolor("#0B0F1A")
        for ax in axes:
            ax.set_facecolor("#111827")
            ax.tick_params(colors="#94A3B8", labelsize=8)
            for spine in ax.spines.values(): spine.set_edgecolor("#1E2A40")

        if "Attrition" in df.columns:
            counts = df["Attrition"].map({"Yes":1,"No":0}).value_counts(normalize=True)*100
            axes[0].bar(["Staying","Leaving"], counts.sort_index(ascending=False).values,
                        color=["#26DE81","#FF4757"])
            axes[0].set_title("Attrition Rate (%)", color="#F1F5F9", fontsize=11)

        if "Department" in df.columns and "Attrition" in df.columns:
            dept_attr = df.groupby("Department")["Attrition"].apply(
                lambda x: (x=="Yes").sum()/len(x)*100).sort_values(ascending=True)
            axes[1].barh(dept_attr.index, dept_attr.values, color="#38BDF8")
            axes[1].set_title("Attrition % by Department", color="#F1F5F9", fontsize=11)
            axes[1].set_xlabel("Attrition %", color="#94A3B8")

        if "JobSatisfaction" in df.columns and "Attrition" in df.columns:
            for val, color, label in [(1,"#FF4757","Left"),(0,"#26DE81","Stayed")]:
                sat_map = df["Attrition"].map({"Yes":1,"No":0}) if df["Attrition"].dtype == object else df["Attrition"]
                subset  = df[sat_map == val]["JobSatisfaction"]
                axes[2].hist(subset, bins=4, alpha=0.7, color=color, label=label, edgecolor="#0B0F1A")
            axes[2].set_title("Job Satisfaction vs Attrition", color="#F1F5F9", fontsize=11)
            axes[2].legend(facecolor="#111827", labelcolor="#94A3B8")
            axes[2].set_xlabel("Satisfaction Score", color="#94A3B8")

        plt.tight_layout()
        st.pyplot(fig)

    # ── TAB 5 — Model Evaluation ──────────────────────────────────────────
    with tabs[4]:
        st.markdown("### Model Comparison")

        if "ret_Xtr" not in st.session_state:
            st.warning("Upload and train in Tab 1 first.")
            st.stop()

        X_tr = st.session_state["ret_Xtr"]
        X_te = st.session_state["ret_Xte"]
        y_tr = st.session_state["ret_ytr"]
        y_te = st.session_state["ret_yte"]

        eval_models = {
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

        for (name, m), color in zip(eval_models.items(), colors):
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_te)
            y_prob = m.predict_proba(X_te)[:, 1]
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
