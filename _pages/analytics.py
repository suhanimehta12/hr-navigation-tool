import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def show():
    st.markdown("""
    <div class='page-header'>
        <h1>📊 Platform Analytics</h1>
        <p>Cross-module insights. See the full employee lifecycle in one view — recruitment quality, retention health, and promotion pipeline.</p>
    </div>
    """, unsafe_allow_html=True)

    has_rec  = "rec_data"  in st.session_state
    has_ret  = "ret_df"    in st.session_state
    has_prom = "prom_df"   in st.session_state

    # ── Status tiles ──────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    for col, has, label, icon, color in [
        (c1, has_rec,  "Recruitment",  "📄", "#38BDF8"),
        (c2, has_ret,  "Retention",    "🔄", "#A5B4FC"),
        (c3, has_prom, "Promotion",    "🏆", "#26DE81"),
    ]:
        status      = "✅ Data Loaded" if has else "⬜ No Data Yet"
        stat_color  = color if has else "#334155"
        with col:
            st.markdown(f"""
            <div class='card' style='text-align:center;border-color:{stat_color};'>
                <div style='font-size:2rem;'>{icon}</div>
                <div style='font-family:Syne,sans-serif;font-weight:700;color:#F1F5F9;margin-top:8px;'>{label}</div>
                <div style='color:{stat_color};font-size:0.85rem;margin-top:6px;font-weight:600;'>{status}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key metrics from all loaded modules ───────────────────────────────
    metrics = []

    if has_rec:
        df_rec = st.session_state["rec_data"]
        hire_rate = (df_rec["HiringDecision"].sum() / len(df_rec) * 100) if "HiringDecision" in df_rec.columns else 0
        metrics.append(("Hire Rate",      f"{hire_rate:.1f}%",   "Recruitment", "#38BDF8"))
        metrics.append(("Total Applicants", f"{len(df_rec):,}", "Recruitment", "#38BDF8"))

    if has_ret:
        df_ret = st.session_state["ret_df"]
        if "Attrition" in df_ret.columns:
            attr_rate = (df_ret["Attrition"] == "Yes").sum() / len(df_ret) * 100
            metrics.append(("Attrition Rate",  f"{attr_rate:.1f}%",   "Retention", "#A5B4FC"))
            metrics.append(("Total Employees", f"{len(df_ret):,}",    "Retention", "#A5B4FC"))

    if has_prom:
        df_prom = st.session_state["prom_df"]
        if "is_promoted" in df_prom.columns:
            prom_rate = df_prom["is_promoted"].mean() * 100
            metrics.append(("Promotion Rate",    f"{prom_rate:.1f}%",  "Promotion", "#26DE81"))
            metrics.append(("Employees Tracked", f"{len(df_prom):,}", "Promotion", "#26DE81"))

    if metrics:
        cols = st.columns(len(metrics))
        for col, (label, value, module, color) in zip(cols, metrics):
            with col:
                st.markdown(f"""
                <div class='metric-tile'>
                    <div class='metric-value' style='color:{color};font-size:1.8rem;'>{value}</div>
                    <div class='metric-label'>{label}</div>
                    <div style='color:#334155;font-size:0.7rem;margin-top:4px;'>{module}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # ── Visual summaries ──────────────────────────────────────────────────
    charts = []

    if has_rec and "HiringDecision" in st.session_state["rec_data"].columns:
        df_rec = st.session_state["rec_data"]
        charts.append(("Hiring Distribution", df_rec["HiringDecision"].value_counts(), "#38BDF8", "bar"))

    if has_ret and "Department" in st.session_state["ret_df"].columns and "Attrition" in st.session_state["ret_df"].columns:
        df_ret = st.session_state["ret_df"]
        dept_attr = df_ret.groupby("Department")["Attrition"].apply(lambda x: (x=="Yes").sum())
        charts.append(("Attrition by Department", dept_attr, "#A5B4FC", "barh"))

    if has_prom and "department" in st.session_state["prom_df"].columns and "is_promoted" in st.session_state["prom_df"].columns:
        df_prom = st.session_state["prom_df"]
        dept_prom = df_prom.groupby("department")["is_promoted"].sum()
        charts.append(("Promotions by Department", dept_prom, "#26DE81", "barh"))

    if charts:
        chart_cols = st.columns(len(charts))
        for col, (title, data, color, chart_type) in zip(chart_cols, charts):
            with col:
                fig, ax = plt.subplots(figsize=(5, 3))
                fig.patch.set_facecolor("#0B0F1A")
                ax.set_facecolor("#111827")
                ax.tick_params(colors="#94A3B8", labelsize=7)
                for spine in ax.spines.values(): spine.set_edgecolor("#1E2A40")

                if chart_type == "bar":
                    ax.bar(data.index.astype(str), data.values, color=color)
                elif chart_type == "barh":
                    ax.barh(data.index.astype(str), data.values, color=color)

                ax.set_title(title, color="#F1F5F9", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
    else:
        st.markdown("""
        <div class='card' style='text-align:center;padding:48px;'>
            <div style='font-size:3rem;margin-bottom:16px;'>📊</div>
            <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:#475569;'>
                No data loaded yet
            </div>
            <div style='color:#334155;font-size:0.9rem;margin-top:8px;'>
                Upload datasets in the Recruitment, Retention, and Promotion modules to see analytics here.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Journey connectivity status ───────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:0.8rem;font-weight:700;color:#334155;
                text-transform:uppercase;letter-spacing:0.12em;margin-bottom:16px;'>
        Employee Journey Connectivity
    </div>
    """, unsafe_allow_html=True)

    all_loaded = has_rec and has_ret and has_prom
    if all_loaded:
        st.success("✅ All three modules loaded — full employee journey intelligence is active.")
    else:
        missing = []
        if not has_rec:  missing.append("Recruitment")
        if not has_ret:  missing.append("Retention")
        if not has_prom: missing.append("Promotion")
        st.warning(f"Load data in: **{', '.join(missing)}** to activate full journey intelligence.")
