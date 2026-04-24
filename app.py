import streamlit as st
import pickle
import numpy as np
from datetime import date

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Fare Predictor",
    page_icon="✈️",
    layout="centered",
)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("flight_rf.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ── Title ───────────────────────────────────────────────────────────────────────
st.title("✈️ Flight Fare Predictor")
st.write("Fill in the flight details below to get an estimated fare price.")
st.divider()

# ── Input Section ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Route & Date")

    airline = st.selectbox("Airline", [
        "Air India", "GoAir", "IndiGo", "Jet Airways",
        "Jet Airways Business", "Multiple carriers",
        "Multiple carriers Premium economy", "SpiceJet",
        "Trujet", "Vistara", "Vistara Premium economy"
    ])

    source = st.selectbox("Source (Departure City)", [
        "Banglore", "Chennai", "Delhi", "Kolkata", "Mumbai"
    ])

    destination = st.selectbox("Destination City", [
        "Banglore", "Cochin", "Delhi", "Hyderabad", "Kolkata", "New Delhi"
    ])

    journey_date = st.date_input(
        "Date of Journey",
        value=date.today(),
        min_value=date.today()
    )

with col2:
    st.subheader("Time & Duration")

    dep_time = st.time_input("Departure Time", value=None)
    arrival_time = st.time_input("Arrival Time", value=None)

    total_stops = st.selectbox("Total Stops", [
        "non-stop", "1 stop", "2 stops", "3 stops", "4 stops"
    ])

    duration_hours = st.number_input("Flight Duration — Hours", min_value=0, max_value=24, value=2)
    duration_minutes = st.number_input("Flight Duration — Minutes", min_value=0, max_value=59, value=0)

st.divider()

# ── Predict button ──────────────────────────────────────────────────────────────
if st.button("🔍 Predict Fare", use_container_width=True, type="primary"):

    # ── Validate time inputs ──────────────────────────────────────────────────
    if dep_time is None or arrival_time is None:
        st.error("Please select both Departure Time and Arrival Time.")
        st.stop()

    # ── Date features ─────────────────────────────────────────────────────────
    journey_day   = journey_date.day
    journey_month = journey_date.month

    # ── Time features ─────────────────────────────────────────────────────────
    dep_hour    = dep_time.hour
    dep_minutes = dep_time.minute
    arr_hour    = arrival_time.hour
    arr_minute  = arrival_time.minute

    # ── Stops encoding (ordinal) ──────────────────────────────────────────────
    stops_map = {"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}
    stops_enc = stops_map[total_stops]

    # ── One-hot encode Airline ────────────────────────────────────────────────
    # drop_first=True drops "Air India" as reference category
    airlines = [
        "Airline_Air India", "Airline_GoAir", "Airline_IndiGo",
        "Airline_Jet Airways", "Airline_Jet Airways Business",
        "Airline_Multiple carriers", "Airline_Multiple carriers Premium economy",
        "Airline_SpiceJet", "Airline_Trujet", "Airline_Vistara",
        "Airline_Vistara Premium economy"
    ]
    airline_enc = {col: 0 for col in airlines}
    key = f"Airline_{airline}"
    if key in airline_enc:
        airline_enc[key] = 1
    # Air India is the dropped reference → all zeros

    # ── One-hot encode Source ─────────────────────────────────────────────────
    # drop_first=True drops "Banglore" as reference category
    sources = ["Source_Chennai", "Source_Delhi", "Source_Kolkata", "Source_Mumbai"]
    source_enc = {col: 0 for col in sources}
    key = f"Source_{source}"
    if key in source_enc:
        source_enc[key] = 1
    # Banglore is the dropped reference → all zeros

    # ── One-hot encode Destination ────────────────────────────────────────────
    # drop_first=True drops "Banglore" as reference category
    destinations = [
        "Destination_Cochin", "Destination_Delhi",
        "Destination_Hyderabad", "Destination_Kolkata", "Destination_New Delhi"
    ]
    dest_enc = {col: 0 for col in destinations}
    key = f"Destination_{destination}"
    if key in dest_enc:
        dest_enc[key] = 1
    # Banglore is the dropped reference → all zeros

    # ── Assemble feature vector (same column order as training X) ─────────────
    features = np.array([[
        stops_enc,
        journey_day,
        journey_month,
        dep_hour,
        dep_minutes,
        arr_hour,
        arr_minute,
        duration_hours,
        duration_minutes,
        airline_enc["Airline_Air India"],
        airline_enc["Airline_GoAir"],
        airline_enc["Airline_IndiGo"],
        airline_enc["Airline_Jet Airways"],
        airline_enc["Airline_Jet Airways Business"],
        airline_enc["Airline_Multiple carriers"],
        airline_enc["Airline_Multiple carriers Premium economy"],
        airline_enc["Airline_SpiceJet"],
        airline_enc["Airline_Trujet"],
        airline_enc["Airline_Vistara"],
        airline_enc["Airline_Vistara Premium economy"],
        source_enc["Source_Chennai"],
        source_enc["Source_Delhi"],
        source_enc["Source_Kolkata"],
        source_enc["Source_Mumbai"],
        dest_enc["Destination_Cochin"],
        dest_enc["Destination_Delhi"],
        dest_enc["Destination_Hyderabad"],
        dest_enc["Destination_Kolkata"],
        dest_enc["Destination_New Delhi"],
    ]])

    # ── Predict ───────────────────────────────────────────────────────────────
    prediction = model.predict(features)[0]

    # ── Display result ────────────────────────────────────────────────────────
    st.success(f"### 💰 Estimated Fare: ₹{prediction:,.0f}")
    st.caption(
        f"**{airline}** · {source} → {destination} · "
        f"{total_stops} · {journey_date.strftime('%d %b %Y')} · "
        f"{dep_time.strftime('%H:%M')} → {arrival_time.strftime('%H:%M')}"
    )

# ── Footer ──────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Powered by a Random Forest model trained on Indian domestic flight data.")