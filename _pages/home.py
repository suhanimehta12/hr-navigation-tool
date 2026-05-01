import streamlit as st


def show():
    st.markdown("""
    <div class='page-header'>
        <h1>Welcome to HR Navigator</h1>
        <p>The only HR platform that follows your employees from the day they applied to the day they leave — and uses that full story to help you hire better, keep people longer, and promote the right ones.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Three pillars ──────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class='card'>
            <div style='font-size:2rem;margin-bottom:12px;'>📄</div>
            <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:#F1F5F9;margin-bottom:8px;'>
                Recruitment & Resume AI
            </div>
            <div style='color:#64748B;font-size:0.88rem;line-height:1.6;'>
                Upload resumes, match against job descriptions, score and rank candidates using Culture DNA fingerprinting of your top performers.
            </div>
            <div style='margin-top:16px;'>
                <span style='background:#0EA5E920;color:#38BDF8;padding:4px 10px;border-radius:20px;font-size:0.75rem;font-weight:600;'>
                    Chapter 1
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='card'>
            <div style='font-size:2rem;margin-bottom:12px;'>🔄</div>
            <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:#F1F5F9;margin-bottom:8px;'>
                Retention & Early Warning
            </div>
            <div style='color:#64748B;font-size:0.88rem;line-height:1.6;'>
                Monitor attrition risk continuously with life event signals, manager impact scoring, and automated alerts with specific recommended actions.
            </div>
            <div style='margin-top:16px;'>
                <span style='background:#6366F120;color:#A5B4FC;padding:4px 10px;border-radius:20px;font-size:0.75rem;font-weight:600;'>
                    Chapter 2
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class='card'>
            <div style='font-size:2rem;margin-bottom:12px;'>🏆</div>
            <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:#F1F5F9;margin-bottom:8px;'>
                Promotion Intelligence
            </div>
            <div style='color:#64748B;font-size:0.88rem;line-height:1.6;'>
                Predict post-promotion success, not just eligibility. Identify who will thrive 18 months after promotion and who you will regret promoting.
            </div>
            <div style='margin-top:16px;'>
                <span style='background:#26DE8120;color:#26DE81;padding:4px 10px;border-radius:20px;font-size:0.75rem;font-weight:600;'>
                    Chapter 3
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Journey flow diagram ───────────────────────────────────────────────
    st.markdown("""
    <div style='background:#0F1628;border:1px solid #1E2A40;border-radius:16px;padding:32px;text-align:center;'>
        <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#94A3B8;
                    text-transform:uppercase;letter-spacing:0.12em;margin-bottom:24px;'>
            The Employee Journey — One Continuous Story
        </div>
        <div style='display:flex;align-items:center;justify-content:center;gap:0;flex-wrap:wrap;'>
            <div style='background:#0EA5E920;border:1px solid #0EA5E960;border-radius:12px;
                        padding:16px 24px;text-align:center;min-width:130px;'>
                <div style='color:#38BDF8;font-size:1.4rem;'>📄</div>
                <div style='color:#F1F5F9;font-weight:600;font-size:0.85rem;margin-top:6px;'>Candidate Applies</div>
                <div style='color:#475569;font-size:0.75rem;margin-top:4px;'>Resume parsed & scored</div>
            </div>
            <div style='color:#334155;font-size:1.4rem;padding:0 8px;'>→</div>
            <div style='background:#6366F120;border:1px solid #6366F160;border-radius:12px;
                        padding:16px 24px;text-align:center;min-width:130px;'>
                <div style='color:#A5B4FC;font-size:1.4rem;'>✅</div>
                <div style='color:#F1F5F9;font-weight:600;font-size:0.85rem;margin-top:6px;'>Gets Hired</div>
                <div style='color:#475569;font-size:0.75rem;margin-top:4px;'>Profile carries forward</div>
            </div>
            <div style='color:#334155;font-size:1.4rem;padding:0 8px;'>→</div>
            <div style='background:#FF475720;border:1px solid #FF475760;border-radius:12px;
                        padding:16px 24px;text-align:center;min-width:130px;'>
                <div style='color:#FF6B81;font-size:1.4rem;'>🔄</div>
                <div style='color:#F1F5F9;font-weight:600;font-size:0.85rem;margin-top:6px;'>Risk Monitored</div>
                <div style='color:#475569;font-size:0.75rem;margin-top:4px;'>Alerts sent to manager</div>
            </div>
            <div style='color:#334155;font-size:1.4rem;padding:0 8px;'>→</div>
            <div style='background:#26DE8120;border:1px solid #26DE8160;border-radius:12px;
                        padding:16px 24px;text-align:center;min-width:130px;'>
                <div style='color:#26DE81;font-size:1.4rem;'>🏆</div>
                <div style='color:#F1F5F9;font-weight:600;font-size:0.85rem;margin-top:6px;'>Promotion Evaluated</div>
                <div style='color:#475569;font-size:0.75rem;margin-top:4px;'>18-month success predicted</div>
            </div>
            <div style='color:#334155;font-size:1.4rem;padding:0 8px;'>→</div>
            <div style='background:#FF9F4320;border:1px solid #FF9F4360;border-radius:12px;
                        padding:16px 24px;text-align:center;min-width:130px;'>
                <div style='color:#FF9F43;font-size:1.4rem;'>🔁</div>
                <div style='color:#F1F5F9;font-weight:600;font-size:0.85rem;margin-top:6px;'>Cycle Continues</div>
                <div style='color:#475569;font-size:0.75rem;margin-top:4px;'>New role re-monitored</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Stats row ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:0.8rem;font-weight:700;color:#334155;
                text-transform:uppercase;letter-spacing:0.12em;margin-bottom:16px;'>
        Why This Platform Pays For Itself
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    stats = [
        ("1.5×", "Cost of a bad hire vs salary"),
        ("3×", "Cost of a failed promotion"),
        ("50%", "Salary cost to replace one employee"),
        ("$1M+", "Saved annually for mid-size firms"),
    ]
    for col, (val, label) in zip([s1, s2, s3, s4], stats):
        with col:
            st.markdown(f"""
            <div class='metric-tile'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Use the sidebar to navigate to any module. Start with **Recruitment** to upload your dataset and begin the employee journey.")
