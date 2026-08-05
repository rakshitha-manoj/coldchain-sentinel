"""
ColdChain Sentinel
Vaccine cold chain shipment risk monitoring & analytics.

Run:  streamlit run app.py
(Run pipeline.py first to generate the underlying data.)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="ColdChain Sentinel",
    page_icon="assets/logo.png" if False else "🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAFE_MIN, SAFE_MAX = 2.0, 8.0

CATEGORY_COLORS = {
    "Safe": "#2e7d32",
    "Low Risk": "#1565c0",
    "Moderate Risk": "#ef6c00",
    "High Risk": "#c62828",
}

# ---------------- Global styling ----------------
st.markdown("""
<style>
    .main > div { padding-top: 1.5rem; }
    [data-testid="stMetric"] {
        background-color: #f7f9fb;
        border: 1px solid #e6e9ec;
        border-radius: 10px;
        padding: 14px 16px 8px 16px;
    }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; color: #555; }
    h1 { font-weight: 700; }
    .subtitle { color: #666; font-size: 1.0rem; margin-top: -10px; }
    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 14px;
        font-size: 0.8rem; font-weight: 600; color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    risk_scores = pd.read_csv("data/shipment_risk_scores.csv")
    leg_risk = pd.read_csv("data/leg_risk_profile.csv")
    readings = pd.read_csv("data/cleaned_readings.csv")
    return risk_scores, leg_risk, readings


risk_scores, leg_risk, readings = load_data()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## 🧊 ColdChain Sentinel")
    st.caption("IoT Analytics Mini Project — MAI507C-4")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Overview", "Simulate New Shipment", "Shipment Explorer", "Route Risk Analysis", "All Shipments"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("**Pipeline**")
    st.markdown("Acquisition → Preprocessing → Spark Route Risk → ML Risk Scoring → Dashboard")
    st.divider()
    st.caption("Built by Rakshitha A. Manoj")

# ---------------- Header ----------------
st.markdown("# 🧊 ColdChain Sentinel")
st.markdown('<p class="subtitle">Real-time vaccine cold chain integrity monitoring and shipment risk analytics</p>', unsafe_allow_html=True)
st.divider()

# ================= PAGE: Overview =================
if page == "Overview":
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Shipments", len(risk_scores))
    col2.metric("High Risk Shipments", int((risk_scores["risk_category"] == "High Risk").sum()))
    col3.metric("Avg Risk Score", f"{risk_scores['risk_score'].mean():.1f} / 100")
    col4.metric("Avg Potency Remaining", f"{risk_scores['estimated_potency_remaining_pct'].mean():.1f}%")

    st.write("")
    left, right = st.columns([3, 2])

    with left:
        st.markdown("#### Risk score by shipment scenario")
        summary = risk_scores.groupby("scenario_label", as_index=False)["risk_score"].mean()
        summary = summary.sort_values("risk_score", ascending=True)
        fig = go.Figure(go.Bar(
            x=summary["risk_score"], y=summary["scenario_label"],
            orientation="h", marker_color="#1565c0",
            text=summary["risk_score"].round(1), textposition="outside"
        ))
        fig.update_layout(height=320, xaxis_title="Average Risk Score", yaxis_title="", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Shipment risk breakdown")
        cat_counts = risk_scores["risk_category"].value_counts().reset_index()
        cat_counts.columns = ["risk_category", "count"]
        fig2 = go.Figure(go.Pie(
            labels=cat_counts["risk_category"], values=cat_counts["count"],
            marker=dict(colors=[CATEGORY_COLORS.get(c, "#999") for c in cat_counts["risk_category"]]),
            hole=0.5
        ))
        fig2.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### About this project")
    st.markdown(
        "This system simulates temperature/humidity telemetry for vaccine shipments across four "
        "route legs (cold storage loading, highway transit, customs hub hold, last-mile delivery), "
        "using an ESP32 + DHT22 model built and validated in Wokwi. Raw telemetry is cleaned and "
        "gap-flagged, processed through Apache Spark to identify which route leg carries the highest "
        "historical risk, then scored using a route-risk-weighted excursion severity model that also "
        "estimates shipment potency loss and data confidence."
    )

# ================= PAGE: Shipment Explorer =================
elif page == "Simulate New Shipment":
    st.markdown("### Configure and run a new shipment simulation")
    st.caption("This generates fresh telemetry and runs it through the same preprocessing, "
               "route-risk weighting, and scoring model live — not a lookup from saved data.")

    LEGS = [
        {"name": "cold_storage_loading", "duration": 30, "temp": 4.0, "hum": 45.0},
        {"name": "highway_transit",      "duration": 60, "temp": 5.0, "hum": 50.0},
        {"name": "customs_hub_hold",     "duration": 45, "temp": 6.0, "hum": 55.0},
        {"name": "last_mile_delivery",   "duration": 30, "temp": 5.0, "hum": 48.0},
    ]
    POTENCY_LOSS_RATE = 0.35

    if "sim_log" not in st.session_state:
        st.session_state.sim_log = []

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        scenario = st.selectbox(
            "Scenario",
            ["Clean Trip", "Brief Excursion", "Sustained Excursion", "Sensor Dropout", "Custom"]
        )
    with col_b:
        affected_leg = st.selectbox(
            "Leg affected (if any)",
            [l["name"] for l in LEGS], index=1
        )
    with col_c:
        seed = st.number_input("Simulation seed", min_value=1, max_value=99999, value=101)

    if scenario == "Custom":
        c1, c2 = st.columns(2)
        with c1:
            custom_severity = st.slider("Excursion severity (°C over safe limit)", 0.0, 10.0, 3.0, 0.5)
        with c2:
            custom_duration_pct = st.slider("Excursion duration (% of leg)", 0, 100, 30, 5)

    run_btn = st.button("▶ Run Simulation", type="primary")

    if run_btn:
        import random as _random
        rng = _random.Random(seed)

        rows = []
        tick = 0
        for leg in LEGS:
            is_affected = leg["name"] == affected_leg
            for t in range(leg["duration"]):
                if scenario == "Sensor Dropout" and is_affected and 10 < t < 25:
                    tick += 1
                    continue

                base = leg["temp"]
                if scenario == "Brief Excursion" and is_affected and 15 < t < 25:
                    base = SAFE_MAX + 3.0
                elif scenario == "Sustained Excursion" and is_affected and t > 10:
                    base = SAFE_MAX + 4.5
                elif scenario == "Custom" and is_affected and t < leg["duration"] * (custom_duration_pct / 100):
                    base = SAFE_MAX + custom_severity

                temp = base + rng.uniform(-0.2, 0.2)
                hum = leg["hum"] + rng.uniform(-3.0, 3.0)
                status = "BREACH" if (temp < SAFE_MIN or temp > SAFE_MAX) else "OK"
                rows.append({"elapsed_sec": tick, "leg": leg["name"], "temp_c": temp,
                             "humidity_pct": hum, "status": status, "data_missing": False})
                tick += 1

        sim_df = pd.DataFrame(rows)

        breach_sec = int((sim_df["status"] == "BREACH").sum())
        excursion = (sim_df["temp_c"] - SAFE_MAX).clip(lower=0) + (SAFE_MIN - sim_df["temp_c"]).clip(lower=0)
        excursion_area = float(excursion.sum())
        max_severity = float(excursion.max())

        leg_weight_map = leg_risk.set_index("leg")["breach_rate_pct"] / 100.0
        anomaly_mult = 1.0 + (1.0 - leg_weight_map.get(affected_leg, 0.0)) if breach_sec > 0 else 0.0
        weighted_score = breach_sec * anomaly_mult

        duration_score = min(100, (breach_sec / 165) * 100 * 3)
        severity_score = min(100, max_severity * 15)
        leg_score = min(100, weighted_score * 2)
        risk_score = round(0.4 * duration_score + 0.4 * severity_score + 0.2 * leg_score, 1)

        if breach_sec == 0:
            risk_cat = "Safe"
        elif risk_score < 25:
            risk_cat = "Low Risk"
        elif risk_score < 55:
            risk_cat = "Moderate Risk"
        else:
            risk_cat = "High Risk"

        potency = round(max(0.0, 100.0 - excursion_area * POTENCY_LOSS_RATE), 1)

        st.session_state.sim_log.insert(0, {
            "scenario": scenario, "affected_leg": affected_leg if scenario != "Clean Trip" else "-",
            "seed": seed, "risk_score": risk_score, "risk_category": risk_cat,
            "potency_remaining_pct": potency, "breach_seconds": breach_sec
        })

        badge_color = CATEGORY_COLORS.get(risk_cat, "#999")
        st.markdown(
            f'<span class="badge" style="background-color:{badge_color}">{risk_cat}</span>',
            unsafe_allow_html=True
        )
        st.write("")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Risk Score", f"{risk_score:.1f} / 100")
        m2.metric("Est. Potency Remaining", f"{potency:.1f}%")
        m3.metric("Breach Duration", f"{breach_sec}s")
        m4.metric("Max Excursion Severity", f"{max_severity:.2f}°C")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sim_df["elapsed_sec"], y=sim_df["temp_c"],
                                  mode="lines", name="Temperature (°C)", line=dict(color="#1565c0", width=2)))
        fig.add_hrect(y0=SAFE_MIN, y1=SAFE_MAX, fillcolor="green", opacity=0.06, line_width=0)
        fig.add_hline(y=SAFE_MAX, line_dash="dash", line_color="#c62828", annotation_text="Safe max (8°C)")
        fig.add_hline(y=SAFE_MIN, line_dash="dash", line_color="#c62828", annotation_text="Safe min (2°C)")
        breach_pts = sim_df[sim_df["status"] == "BREACH"]
        if len(breach_pts) > 0:
            fig.add_trace(go.Scatter(x=breach_pts["elapsed_sec"], y=breach_pts["temp_c"],
                                      mode="markers", name="Breach", marker=dict(color="#c62828", size=7)))
        fig.update_layout(title="Live simulated shipment temperature", height=420,
                           xaxis_title="Elapsed time (s)", yaxis_title="Temperature (°C)")
        st.plotly_chart(fig, use_container_width=True)

    if st.session_state.sim_log:
        st.markdown("#### Simulation session log")
        st.dataframe(pd.DataFrame(st.session_state.sim_log), use_container_width=True, hide_index=True)


    st.markdown("### Inspect a single shipment run")

    run_ids = sorted(risk_scores["run_id"].unique())
    selected_run = st.selectbox("Select shipment run", run_ids, format_func=lambda r: f"Run {r}")

    run_info = risk_scores[risk_scores["run_id"] == selected_run].iloc[0]
    run_readings = readings[readings["run_id"] == selected_run].sort_values("elapsed_sec")

    badge_color = CATEGORY_COLORS.get(run_info["risk_category"], "#999")
    st.markdown(
        f'<span class="badge" style="background-color:{badge_color}">{run_info["risk_category"]}</span> '
        f'&nbsp; <b>{run_info["scenario_label"]}</b>',
        unsafe_allow_html=True
    )
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Score", f"{run_info['risk_score']:.1f} / 100")
    c2.metric("Est. Potency Remaining", f"{run_info['estimated_potency_remaining_pct']:.1f}%")
    c3.metric("Data Confidence", f"{run_info['data_confidence_pct']:.1f}%")
    c4.metric("Breach Duration", f"{int(run_info['total_breach_seconds'])}s")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=run_readings["elapsed_sec"], y=run_readings["temp_c"],
        mode="lines", name="Temperature (°C)", line=dict(color="#1565c0", width=2)
    ))
    fig.add_hrect(y0=SAFE_MIN, y1=SAFE_MAX, fillcolor="green", opacity=0.06, line_width=0)
    fig.add_hline(y=SAFE_MAX, line_dash="dash", line_color="#c62828", annotation_text="Safe max (8°C)")
    fig.add_hline(y=SAFE_MIN, line_dash="dash", line_color="#c62828", annotation_text="Safe min (2°C)")

    breach_points = run_readings[run_readings["status"] == "BREACH"]
    if len(breach_points) > 0:
        fig.add_trace(go.Scatter(
            x=breach_points["elapsed_sec"], y=breach_points["temp_c"],
            mode="markers", name="Breach", marker=dict(color="#c62828", size=7)
        ))

    missing_points = run_readings[run_readings["data_missing"] == True]
    if len(missing_points) > 0:
        fig.add_trace(go.Scatter(
            x=missing_points["elapsed_sec"], y=missing_points["temp_c"],
            mode="markers", name="Interpolated (sensor dropout)",
            marker=dict(color="gray", size=7, symbol="x")
        ))

    fig.update_layout(
        title=f"Run {selected_run} — Temperature across shipment journey",
        xaxis_title="Elapsed time (seconds)", yaxis_title="Temperature (°C)",
        height=460, legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Route legs: 0–29s Cold Storage Loading · 30–89s Highway Transit · 90–134s Customs Hub Hold · 135–164s Last-Mile Delivery")

# ================= PAGE: Route Risk Analysis =================
elif page == "Route Risk Analysis":
    st.markdown("### Which route leg is historically riskiest?")
    st.caption("Aggregated across all 20 simulated shipment runs using Apache Spark distributed processing")

    leg_sorted = leg_risk.sort_values("breach_rate_pct", ascending=True)
    fig2 = go.Figure(go.Bar(
        x=leg_sorted["breach_rate_pct"], y=leg_sorted["leg"],
        orientation="h", marker_color="#c62828",
        text=leg_sorted["breach_rate_pct"].astype(str) + "%", textposition="outside"
    ))
    fig2.update_layout(title="Breach rate by route leg", xaxis_title="Breach rate (%)", yaxis_title="", height=380)
    st.plotly_chart(fig2, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig3 = go.Figure(go.Bar(
            x=leg_risk["leg"], y=leg_risk["avg_excursion_severity_c"],
            marker_color="#ef6c00"
        ))
        fig3.update_layout(title="Avg excursion severity (°C over safe limit)", height=320)
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        fig4 = go.Figure(go.Bar(
            x=leg_risk["leg"], y=leg_risk["dropout_rate_pct"],
            marker_color="#616161"
        ))
        fig4.update_layout(title="Sensor dropout rate by leg", height=320)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### Full leg risk profile")
    st.dataframe(leg_risk, use_container_width=True, hide_index=True)

# ================= PAGE: All Shipments =================
elif page == "All Shipments":
    st.markdown("### All 20 shipment runs")

    filter_cat = st.multiselect(
        "Filter by risk category",
        options=risk_scores["risk_category"].unique().tolist(),
        default=risk_scores["risk_category"].unique().tolist()
    )
    filtered = risk_scores[risk_scores["risk_category"].isin(filter_cat)]

    def color_category(val):
        color = CATEGORY_COLORS.get(val, "#999")
        return f"background-color: {color}20; color: {color}; font-weight: 600;"

    st.dataframe(
        filtered.style.map(color_category, subset=["risk_category"]),
        use_container_width=True,
        hide_index=True
    )
