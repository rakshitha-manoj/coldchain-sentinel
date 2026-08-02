# ColdChain Sentinel

**IoT Analytics mini-application — vaccine cold chain shipment risk monitoring**

Simulates temperature/humidity telemetry for vaccine shipments across four route
legs (cold storage loading, highway transit, customs hub hold, last-mile delivery),
processes it through a route-risk-weighted excursion severity model, and estimates
shipment potency loss and data confidence — presented as a live interactive
dashboard.

Built as part of the IoT Analytics course (MAI507C-4) CIA-3 mini-project.

## Live demo

Deployed on Streamlit Community Cloud: (add your link here after deploying)

## Pipeline

Wokwi (ESP32 + DHT22 simulation) -> Preprocessing (gap detection + interpolation)
-> Apache Spark (distributed route-leg risk profiling) -> Risk Modeling (excursion
severity + potency/TOR estimate) -> This dashboard

## Key finding

Spark-based aggregation across 20 simulated shipment runs identifies **highway
transit** as the highest-risk leg (16.33% breach rate) - higher than customs hub
hold (4.0%), despite customs hold being the leg most commonly assumed riskiest.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

Author: Rakshitha A. Manoj
