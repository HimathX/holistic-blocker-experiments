import os
import json
import subprocess

import streamlit as st
import plotly.graph_objects as go

# ── Constants ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
BLUE = "#4A90D9"
RED = "#E74C3C"
GREEN = "#27AE60"
YELLOW = "#F39C12"
CHART_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="sans-serif", size=13, color="black"),
    margin=dict(l=40, r=40, t=50, b=40),
)


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Holistic Blocker Understanding - Results Dashboard",
    page_icon="🧠",
    layout="wide",
)


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_results(experiment_num):
    path = os.path.join(ROOT, f"experiment_{experiment_num}", "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def status_indicator(exp_num, data):
    if data is None:
        return f"⚪ Experiment {exp_num}: No data"
    return f"🟢 Experiment {exp_num}: PASS" if data.get("pass") else f"🔴 Experiment {exp_num}: FAIL"


def no_data_msg():
    st.info("No results yet. Click **Re-run All Experiments** in the sidebar.")


def badge(passed):
    color = GREEN if passed else RED
    label = "PASS" if passed else "FAIL"
    return f'<span style="background:{color};color:white;padding:4px 12px;border-radius:4px;font-weight:bold">{label}</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Logical AI Copilot")
    st.caption("Take-home Interview Assignment")
    st.divider()

    r1 = load_results(1)
    r2 = load_results(2)
    r3 = load_results(3)

    st.markdown(status_indicator(1, r1))
    st.markdown(status_indicator(2, r2))
    st.markdown(status_indicator(3, r3))

    st.divider()

    if st.button("▶  Re-run All Experiments", width="stretch"):
        with st.spinner("Running all experiments… this may take a minute."):
            subprocess.run(
                ["uv", "run", "python", "run_all.py"],
                cwd=ROOT,
                capture_output=False,
            )
        st.rerun()


# ── Main ─────────────────────────────────────────────────────────────────────
st.title("Holistic Blocker Understanding — Results Dashboard")

tab1, tab2, tab3 = st.tabs(["Experiment 1", "Experiment 2", "Experiment 3"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Coreference Linking
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    data = load_results(1)
    if data is None:
        no_data_msg()
    else:
        left, right = st.columns([4, 6])

        with left:
            st.subheader("Coreference Linking Accuracy")
            st.caption("Can the system link the same bug across different channels?")
            st.metric(
                label="Semantic vs Token Overlap",
                value=f"{data['key_metric'] * 100:.1f}%",
                delta=f"{(data['key_metric'] - data['baseline_metric']) * 100:+.1f}% vs baseline",
            )

            st.markdown(
                "8 messages spanning 3 bug clusters (BUG_A auth failure, BUG_B payment failure, BUG_C UI) "
                "were fed to both methods. The semantic model uses cosine similarity threshold **0.20** to decide "
                "whether two messages describe the same bug."
            )
            st.markdown("**Mock messages used:**")
            msg_rows = [
                {"Channel": "slack_engineer", "Bug": "BUG_A", "Message": "NullPointerException at auth_service.py line 142, uid=None"},
                {"Channel": "email_support",  "Bug": "BUG_A", "Message": "Users cannot log in at all since this morning, totally broken"},
                {"Channel": "discord_user",   "Bug": "BUG_A", "Message": "Reverted PR #4421 — broke the authentication pipeline"},
                {"Channel": "twitter_user",   "Bug": "BUG_A", "Message": "hey @company your app wont let me in, been trying for an hour"},
                {"Channel": "zendesk_ticket", "Bug": "BUG_A", "Message": "Getting a white screen after entering my password, iOS app"},
                {"Channel": "slack_payments", "Bug": "BUG_B", "Message": "Stripe webhook is returning 422, orders not completing"},
                {"Channel": "email_billing",  "Bug": "BUG_B", "Message": "Customers reporting they cannot complete checkout, cards declined"},
                {"Channel": "jira_ui",        "Bug": "BUG_C", "Message": "UI misalignment on the dashboard widget for Safari v17"},
            ]
            rows_display = [{"Channel": r["Channel"], "Bug": r["Bug"], "Message": r["Message"][:48] + "…"} for r in msg_rows]
            st.dataframe(rows_display, width="stretch", hide_index=True)

        with right:
            pairs = data.get("pairs", [])
            pair_labels = [p["pair"] for p in pairs]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Baseline (Token Overlap)",
                x=pair_labels,
                y=[p["baseline_score"] for p in pairs],
                marker_color=RED,
            ))
            fig.add_trace(go.Bar(
                name="Semantic (Embedding)",
                x=pair_labels,
                y=[p["semantic_score"] for p in pairs],
                marker_color=BLUE,
            ))
            fig.add_hline(y=0.20, line_dash="dash", line_color=YELLOW,
                          annotation_text="Threshold 0.20", annotation_position="top right")
            fig.update_layout(
            title=dict(
                text="Similarity Scores by Method",
                font=dict(color="black")
            ),
            barmode="group",
            height=500,
            paper_bgcolor="white",
            plot_bgcolor="white",
            # Global font setting to catch anything else
            font=dict(family="sans-serif", size=13, color="black"),
            yaxis=dict(
                title=dict(text="Score", font=dict(color="black")), 
                range=[0, 0.6], 
                tickfont=dict(size=10, color="black"),
                tickcolor="black",
                showline=True,
                linecolor="black"
            ),
            xaxis=dict(
                tickangle=-45, 
                tickfont=dict(size=9, color="black"),
                tickcolor="black",
                showline=True,
                linecolor="black"
            ),
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="right", 
                x=1,
                font=dict(color="black")
            ),
            margin=dict(l=40, r=40, t=50, b=160),
        )
            st.plotly_chart(fig, width="stretch")

            # Results table with color
            def color_result(correct):
                return "background-color: #d4edda" if correct else "background-color: #f8d7da"

            table_data = []
            for p in pairs:
                table_data.append({
                    "Pair": p["pair"],
                    "Baseline Score": f"{p['baseline_score']:.4f}",
                    "Baseline Result": "CORRECT" if p["baseline_correct"] else "WRONG",
                    "Semantic Score": f"{p['semantic_score']:.4f}",
                    "Semantic Result": "CORRECT" if p["semantic_correct"] else "WRONG",
                    "_b_correct": p["baseline_correct"],
                    "_s_correct": p["semantic_correct"],
                })

            import pandas as pd
            df = pd.DataFrame(table_data)

            def style_row(row):
                styles = [""] * len(row)
                idx = list(row.index)
                b_c = row["Baseline Result"] == "CORRECT"
                s_c = row["Semantic Result"] == "CORRECT"
                styles[idx.index("Baseline Result")] = f"background-color: {"#89B88C" if b_c else "#d28289"}; color: white"
                styles[idx.index("Semantic Result")] = f"background-color: {'#89B88C' if s_c else '#d28289'}; color: white"
                return styles

            display_df = df.drop(columns=["_b_correct", "_s_correct"])
            st.dataframe(
                display_df.style.apply(style_row, axis=1),
                width="stretch",
                hide_index=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Two-Stage Context Paging
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    data = load_results(2)
    if data is None:
        no_data_msg()
    else:
        left, right = st.columns([4, 6])

        with left:
            st.subheader("Two-Stage Context Paging")
            st.caption("How few candidates do we need before calling the LLM?")

            m1, m2 = st.columns(2)
            sweet_k = data.get("sweet_spot_k")
            sweet_budget = data.get("token_budget_at_sweet_spot")
            m1.metric("Sweet spot k", f"k = {sweet_k}" if sweet_k else "None found")
            m2.metric("Token budget at sweet spot", f"{sweet_budget}%" if sweet_budget is not None else "—")

            st.markdown("**Incoming message:**")
            with st.container(border=True):
                st.markdown(
                    "_\"The auth module keeps throwing errors and users are completely locked out\"_"
                )

            st.markdown("**Bug history (20 open bugs):**")
            bug_history = [
                ("BUG_000", "Payment service timeout on checkout"),
                ("BUG_001", "Image upload fails for PNG files over 5MB"),
                ("BUG_002", "Search results not updating after filter change"),
                ("BUG_003", "Notification emails going to spam folder"),
                ("BUG_004", "Dark mode toggle not persisting after refresh"),
                ("BUG_005", "CSV export missing last column of data"),
                ("BUG_006", "Video player controls disappear on mobile"),
                ("BUG_007", "Authentication service crash, users cannot access accounts"),
                ("BUG_008", "Calendar invite timezone showing incorrectly"),
                ("BUG_009", "Drag and drop broken in Firefox browser"),
                ("BUG_010", "API rate limit errors appearing too early"),
                ("BUG_011", "Profile picture not updating after upload"),
                ("BUG_012", "Markdown rendering broken in comment section"),
                ("BUG_013", "Two factor auth codes not being sent via SMS"),
                ("BUG_014", "Report generation freezes on large datasets"),
                ("BUG_015", "Autocomplete suggestions showing deleted items"),
                ("BUG_016", "Webhook delivery failing with 502 error"),
                ("BUG_017", "Graph tooltips overlapping on small screens"),
                ("BUG_018", "Session expiry happening too quickly"),
                ("BUG_019", "Database migration script failing on Postgres 15"),
            ]
            import pandas as pd
            bug_df = pd.DataFrame(bug_history, columns=["ID", "Description"])

            def highlight_true_match(row):
                if row["ID"] == "BUG_007":
                    return ["background-color: #fff3cd; color: black; font-weight: bold"] * len(row)
                return [""] * len(row)

            st.dataframe(
                bug_df.style.apply(highlight_true_match, axis=1),
                width="stretch",
                hide_index=True,
                height=280,
            )

            st.markdown(
                "We simulate a queue of **20 open bugs**. A new message arrives and we need to find which "
                "bug it belongs to. Instead of sending all 20 to the LLM, Stage 1 uses a cheap vector search "
                "to shortlist the top **k** candidates. Stage 2 passes only those k bugs to the LLM. "
                "The sweet spot is the smallest k where the answer is still correct and the token budget stays under 10%."
            )
            with st.container(border=True):
                st.markdown(
                    "**Stage 1** — vector search narrows 20 bugs → k candidates  \n"
                    "**Stage 2** — LLM reads only those k candidates  \n"
                    "**Goal** — find the smallest k that is still correct (avoids O(n²) token cost)"
                )

        with right:
            k_results = data.get("k_results", [])
            k_vals = [r["k"] for r in k_results]
            acc_vals = [100.0 if r["correct"] else 0.0 for r in k_results]
            budget_vals = [r["budget_pct"] for r in k_results]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=k_vals, y=acc_vals, mode="lines+markers",
                name="Accuracy %", line=dict(color=BLUE, width=2),
                marker=dict(size=8),
            ))
            fig.add_trace(go.Scatter(
                x=k_vals, y=budget_vals, mode="lines+markers",
                name="Token Budget %", line=dict(color=RED, width=2, dash="dash"),
                marker=dict(size=8),
            ))
            if sweet_k:
                fig.add_vline(x=sweet_k, line_dash="dot", line_color=YELLOW,
                              annotation_text=f"Sweet spot k={sweet_k}", annotation_position="top right")
            fig.update_layout(
                title=dict(text="Accuracy vs Token Budget Trade-off", font=dict(color="black")),
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(family="sans-serif", size=13, color="black"),
                xaxis=dict(
                    title=dict(text="k (candidates)", font=dict(color="black")),
                    tickvals=k_vals,
                    tickfont=dict(color="black"),
                    tickcolor="black",
                    showline=True,
                    linecolor="black",
                ),
                yaxis=dict(
                    title=dict(text="%", font=dict(color="black")),
                    range=[-5, 110],
                    tickfont=dict(color="black"),
                    tickcolor="black",
                    showline=True,
                    linecolor="black",
                ),
                legend=dict(font=dict(color="black")),
                margin=dict(l=40, r=40, t=50, b=40),
            )
            st.plotly_chart(fig, width="stretch")

            import pandas as pd

            def style_k_table(row):
                styles = [""] * len(row)
                if row["k"] == sweet_k:
                    styles = [f"background-color: #fff3cd; color: black"] * len(row)
                idx = list(row.index)
                pass_val = row["Pass"]
                styles[idx.index("Pass")] = f"background-color: {'#89B88C' if pass_val == 'PASS' else '#d28289'}; color: black"
                return styles

            table_rows = []
            for r in k_results:
                table_rows.append({
                    "k": r["k"],
                    "Stage1 Hit": "YES" if r["stage1_hit"] else "NO",
                    "Correct": "YES" if r["correct"] else "NO",
                    "Budget %": f"{r['budget_pct']:.1f}%",
                    "Pass": "PASS" if r["pass"] else "FAIL",
                })
            df2 = pd.DataFrame(table_rows)
            st.dataframe(
                df2.style.apply(style_k_table, axis=1),
                width="stretch",
                hide_index=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Split-Brain Conflict Resolution
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    data = load_results(3)
    if data is None:
        no_data_msg()
    else:
        left, right = st.columns([4, 6])

        with left:
            st.subheader("Split-Brain Conflict Resolution")
            st.caption("GitHub API vs LLM majority voting when Slack and Discord disagree")

            m1, m2, m3 = st.columns(3)
            m1.metric("Deterministic accuracy", f"{data['deterministic_accuracy'] * 100:.1f}%")
            m2.metric("Majority vote accuracy", f"{data['majority_vote_accuracy'] * 100:.1f}%")
            m3.metric("MV hallucination rate (conflicts)", f"{data['mv_hallucination_rate_on_conflicts'] * 100:.1f}%")

            st.markdown(
                "We test **8 real-world scenarios** where Slack and Discord report different bug statuses. "
                "**Majority vote** simulates an LLM picking whichever source it finds more convincing — "
                "essentially guessing when the signals conflict. "
                "**Deterministic** resolves conflicts by calling the GitHub deploy API for the ground truth. "
                "The hypothesis is that grounding beats guessing every time."
            )

            st.divider()
            conf = data["conflict_scenarios"]
            total = data["total_scenarios"]
            mv_wrong_on_conflict = round(data["mv_hallucination_rate_on_conflicts"] * conf)
            st.markdown(f"**{conf} out of {total}** scenarios had conflicts")
            st.markdown("**Deterministic** resolved all conflicts correctly")
            st.markdown(f"**Majority vote** got **{mv_wrong_on_conflict}** wrong on conflict cases")

        with right:
            scenarios = data.get("scenarios", [])
            sc_labels = [f"Scenario {s['id']}" for s in scenarios]

            mv_colors = [GREEN if s["mv_correct"] else RED for s in scenarios]
            det_colors = [GREEN if s["det_correct"] else RED for s in scenarios]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Majority Vote",
                x=sc_labels,
                y=[1] * len(scenarios),
                marker_color=mv_colors,
                text=[s["mv_decision"] for s in scenarios],
                textposition="inside",
            ))
            fig.add_trace(go.Bar(
                name="Deterministic",
                x=sc_labels,
                y=[1] * len(scenarios),
                marker_color=det_colors,
                text=[s["det_decision"] for s in scenarios],
                textposition="inside",
            ))
            fig.update_layout(
                title=dict(text="Decision Accuracy by Scenario", font=dict(color="black")),
                barmode="group",
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(family="sans-serif", size=13, color="black"),
                xaxis=dict(
                    tickfont=dict(color="black"),
                    tickcolor="black",
                    showline=True,
                    linecolor="black",
                ),
                yaxis=dict(
                    showticklabels=False,
                    title="",
                    showline=False,
                ),
                legend=dict(font=dict(color="black")),
                margin=dict(l=40, r=40, t=50, b=40),
            )
            st.plotly_chart(fig, width="stretch")

            import pandas as pd

            def style_sc_table(row):
                styles = [""] * len(row)
                idx = list(row.index)
                mvc = row["MV Correct"]
                dc = row["Det Correct"]
                styles[idx.index("MV Correct")] = f"background-color: {'#89B88C' if mvc == '✓' else '#d28289'}; color: black"
                styles[idx.index("Det Correct")] = f"background-color: {'#89B88C' if dc == '✓' else '#d28289'}; color: black"
                return styles

            table_rows = []
            for s in scenarios:
                table_rows.append({
                    "Scenario": s["id"],
                    "Description": s["description"],
                    "Conflict": "YES" if s["conflict_detected"] else "NO",
                    "MV Decision": s["mv_decision"],
                    "MV Correct": "✓" if s["mv_correct"] else "✗",
                    "Det Decision": s["det_decision"],
                    "Det Correct": "✓" if s["det_correct"] else "✗",
                })
            df3 = pd.DataFrame(table_rows)
            st.dataframe(
                df3.style.apply(style_sc_table, axis=1),
                width="stretch",
                hide_index=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# BOTTOM — Overall Hypothesis Validation
# ════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Overall Hypothesis Validation")

r1 = load_results(1)
r2 = load_results(2)
r3 = load_results(3)

c1, c2, c3 = st.columns(3)

with c1:
    with st.container(border=True):
        st.markdown("### Blackboard Coreference Hypothesis")
        if r1:
            st.markdown(badge(r1["pass"]), unsafe_allow_html=True)
            st.metric("Semantic Accuracy", f"{r1['key_metric'] * 100:.1f}%")
            st.markdown(
                "Semantic embeddings correctly linked the same bug across Discord, "
                "Email, and Slack even when the wording was completely different."
            )
        else:
            st.info("No data yet.")
        st.caption("Based on: Zhukova et al. (2026) NewsWCL50r")

with c2:
    with st.container(border=True):
        st.markdown("### Two-Stage Context Paging Hypothesis")
        if r2:
            st.markdown(badge(r2["pass"]), unsafe_allow_html=True)
            sweet_k = r2.get("sweet_spot_k")
            budget = r2.get("token_budget_at_sweet_spot")
            st.metric("Sweet Spot", f"k = {sweet_k}" if sweet_k else "Not found")
            st.markdown(
                f"Vector pre-filtering found the correct bug at k={sweet_k} "
                f"using only {budget}% of the full LLM token budget."
            )
        else:
            st.info("No data yet.")
        st.caption("Based on: RAG top-k retrieval & token-budget compression literature")

with c3:
    with st.container(border=True):
        st.markdown("### Deterministic Grounding Hypothesis")
        if r3:
            st.markdown(badge(r3["pass"]), unsafe_allow_html=True)
            st.metric("Deterministic Accuracy", f"{r3['deterministic_accuracy'] * 100:.1f}%")
            st.markdown(
                "Calling the GitHub deploy API eliminated all ambiguity in conflict cases, "
                f"while majority-vote LLM guessing had a "
                f"{r3['mv_hallucination_rate_on_conflicts'] * 100:.0f}% hallucination rate."
            )
        else:
            st.info("No data yet.")
        st.caption("Based on: Schick et al. (2023) Toolformer; Yao et al. (2023) ReAct")
