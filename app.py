"""Serve the model saved by train_model.ipynb; never fit a model at app startup."""
from pathlib import Path
import json

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Scoop Forecast | Ice cream sales", page_icon="🍦", layout="centered")


@st.cache_resource
def load_models():
    return joblib.load(ROOT / "models" / "ice_cream_model.joblib")


@st.cache_data
def load_metadata():
    return json.loads((ROOT / "models" / "metadata.json").read_text())


bundle = load_models()
meta = load_metadata()
model = bundle["model"]
baseline = bundle["baseline"]

st.markdown("""
<style>
.stApp {background: #fcfaf5;}
.block-container {max-width: 960px; padding-top: 2.2rem;}
h1, h2, h3 {color: #203d34;}
div[data-testid="stMetric"] {background: #edf4ee; border-radius: 16px; padding: 18px;}
div[data-testid="stMetricValue"] {color: #214d3d;}
</style>
""", unsafe_allow_html=True)

st.caption("SCOOP FORECAST  /  ML FOR BUSINESS APPLICATIONS")
st.title("A little foresight. A lot of scoops. 🍦")
st.write("Explore how temperature and the day of the week change predicted daily ice cream sales.")
st.info("Classroom demo · All training data are simulated. Predictions illustrate a model, rather than actual shop demand.")

forecast_tab, model_tab = st.tabs(["🍦 Predict sales", "📓 Model & notebook"])

with forecast_tab:
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.subheader("Set the day")
        temperature = st.slider("Temperature (°F)", 50, 100, 80, 1,
                                help="Daily temperature; keep inputs within the simulated data range.")
        day_type = st.selectbox("Day type", ["Weekday", "Weekend"],
                               help="New predictor: Saturday or Sunday = weekend (1); other days = weekday (0).")
        st.caption("New predictor: **weekend status**. The simulation assumes that weekends attract more customers at the same temperature.")
    is_weekend = int(day_type == "Weekend")
    inputs = pd.DataFrame({"temperature_f": [float(temperature)], "is_weekend": [is_weekend]})
    prediction = max(0.0, float(model.predict(inputs[bundle["feature_names"]])[0]))
    temp_only = max(0.0, float(baseline.predict(inputs[["temperature_f"]])[0]))
    with right:
        st.subheader("Your forecast")
        st.metric("Predicted daily sales", f"{prediction:,.0f} scoops")
        st.caption(f"{day_type} · {temperature}°F · simulated shop")
        st.write(f"The temperature-only model estimates **{temp_only:,.0f} scoops**. Adding day type changes this estimate by **{prediction - temp_only:+.0f} scoops**.")
        st.caption("This model comparison is not a causal effect or an uncertainty range.")

    st.divider()
    st.subheader("Same weather. Different day.")
    st.caption("Predictions from the saved model. The points mark your selected temperature.")
    temperatures = np.arange(50, 101)
    curves = []
    for weekend, label in [(0, "Weekday"), (1, "Weekend")]:
        frame = pd.DataFrame({"temperature_f": temperatures, "is_weekend": weekend})
        frame["Predicted scoops"] = model.predict(frame[bundle["feature_names"]]).clip(min=0)
        frame["Day"] = label
        curves.append(frame)
    chart_data = pd.concat(curves, ignore_index=True)
    encoding = dict(
        x=alt.X("temperature_f:Q", title="Temperature (°F)", scale=alt.Scale(domain=[50, 100])),
        y=alt.Y("Predicted scoops:Q", title="Scoops / day", scale=alt.Scale(zero=True)),
        color=alt.Color("Day:N", scale=alt.Scale(domain=["Weekday", "Weekend"], range=["#487d68", "#dd9164"])),
        tooltip=[alt.Tooltip("Day:N"), alt.Tooltip("temperature_f:Q", title="Temperature (°F)"),
                 alt.Tooltip("Predicted scoops:Q", format=".0f")],
    )
    lines = alt.Chart(chart_data).mark_line(strokeWidth=3).encode(**encoding)
    points = alt.Chart(chart_data[chart_data["temperature_f"] == temperature]).mark_point(size=95, filled=True).encode(**encoding)
    st.altair_chart((lines + points).properties(height=260), width="stretch")

with model_tab:
    st.subheader("What changed")
    st.write("A temperature-only linear regression is the baseline. The updated regression adds **is_weekend**, encoded as 0 for weekdays and 1 for weekends. Both are trained in the notebook; this app loads the saved models to make predictions.")
    st.subheader("Evaluation on held-out simulated days")
    rows = []
    for key, label in [("baseline", "Temperature only"), ("updated", "Temperature + weekend")]:
        m = meta["evaluation"][key]
        rows.append({"Model": label, "MAE (scoops)": round(m["mae"], 2),
                     "RMSE (scoops)": round(m["rmse"], 2), "R²": round(m["r2"], 3)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Chronological split: first {meta['train_rows']} days for training, last {meta['test_rows']} for testing. MAE is average absolute prediction error; lower is better. Both models use the same split. The served models are refit on all {meta['total_rows']} simulated days after evaluation.")
    st.write("**Data source:** A reproducible classroom simulation, not observed business records. The notebook uses a fixed random seed and explicitly builds temperature, weekend status, and sales. The comparison demonstrates recovery of a pattern deliberately included in that simulation; it does not validate real-world demand or a causal weekend effect.")
    st.write("**Target:** Scoops sold per day. **Predictors:** Temperature in °F and weekend status. **Model:** scikit-learn linear regression.")
    st.subheader("Inspect and reproduce")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download executed notebook", (ROOT / "train_model.ipynb").read_bytes(),
                           file_name="train_model.ipynb", mime="application/x-ipynb+json", width="stretch")
        st.download_button("Download simulated data", (ROOT / "data" / "ice_cream_sales.csv").read_bytes(),
                           file_name="ice_cream_sales.csv", mime="text/csv", width="stretch")
    with col2:
        st.download_button("Download saved model", (ROOT / "models" / "ice_cream_model.joblib").read_bytes(),
                           file_name="ice_cream_model.joblib", mime="application/octet-stream", width="stretch")
        st.link_button("View source on GitHub", "https://github.com/ymohamed-andrewcmu/assignment-3-ice-cream", width="stretch")
    st.caption("Run the notebook from top to bottom to regenerate the data, evaluate both models, refit and save, then verify a reloaded prediction. There is no model training in the app.")

st.caption("Scoop Forecast · Assignment 3 · Yaseen Mohamed")
