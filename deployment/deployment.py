import streamlit as st
import pandas as pd
import joblib

# Load trained model
MODEL_PATH = r"F:\Courses\DEPI\Data Scientist\finalproject\Pre2Predict\Model\trained_model.pkl"
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Sales Prediction", layout="centered")
st.title("🛒 Sales Prediction for Retail Stores")
st.markdown("Enter store features below to predict expected sales.")

# --- User Input Function ---
def user_input_features():
    Store = st.number_input("Store ID", min_value=1, step=1)
    
    StoreType_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    StoreType = StoreType_map[st.selectbox("Store Type", list(StoreType_map.keys()))]

    CompetitionDistance = st.number_input("Competition Distance (meters)", min_value=0.0)

    Promo2 = st.radio("Promo2 Active?", ["No", "Yes"]) == "Yes"
    Promo2SinceWeek = st.number_input("Promo2 Since Week (0 if not active)", min_value=0, max_value=52, step=1)
    Promo2SinceYear = st.number_input("Promo2 Since Year (0 if not active)", min_value=1900, max_value=2100)

    Customers = st.number_input("Expected Number of Customers", min_value=0)
    Open = st.radio("Is the store open?", ["Closed", "Open"]) == "Open"
    Promo = st.radio("Is a Promo running?", ["No", "Yes"]) == "Yes"
    SchoolHoliday = st.radio("Is it a school holiday?", ["No", "Yes"]) == "Yes"

    Date_year = st.slider("Year", 2013, 2025, 2022)
    Date_month = st.slider("Month", 1, 12, 6)
    Date_day = st.slider("Day", 1, 31, 15)

    # Assortment
    assortment_choice = st.selectbox("Assortment Type", ["Basic", "Extra", "Extended"])
    Assortment_0 = assortment_choice == "Basic"
    Assortment_1 = assortment_choice == "Extra"
    Assortment_2 = assortment_choice == "Extended"

    # State Holiday
    StateHoliday_0 = st.radio("Is there a state holiday?", ["No", "Yes"]) == "No"

    # PromoInterval
    promo_interval = st.selectbox("Promo Interval", ["None", "Jan-Apr", "May-Aug", "Sept-Dec"])
    PromoInterval_0 = promo_interval == "None"
    PromoInterval_1 = promo_interval == "Jan-Apr"
    PromoInterval_2 = promo_interval == "May-Aug"
    PromoInterval_3 = promo_interval == "Sept-Dec"

    # Day of Week
    dow_map = {
        "Monday": 1, "Tuesday": 2, "Wednesday": 3,
        "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7
    }
    dow_choice = st.selectbox("Day of the Week", list(dow_map.keys()))
    day_of_week_encoded = {f"DayOfWeek_{i}": (dow_map[dow_choice] == i) for i in range(1, 8)}

    # Final data dictionary
    data = {
        'Store': Store,
        'StoreType': StoreType,
        'CompetitionDistance': CompetitionDistance,
        'Promo2': int(Promo2),
        'Promo2SinceWeek': Promo2SinceWeek,
        'Promo2SinceYear': Promo2SinceYear,
        'Customers': Customers,
        'Open': int(Open),
        'Promo': int(Promo),
        'SchoolHoliday': int(SchoolHoliday),
        'Date_year': Date_year,
        'Date_month': Date_month,
        'Date_day': Date_day,
        'PromoInterval_0': PromoInterval_0,
        'PromoInterval_1': PromoInterval_1,
        'PromoInterval_2': PromoInterval_2,
        'PromoInterval_3': PromoInterval_3,
        'StateHoliday_0': StateHoliday_0,
        'Assortment_0': Assortment_0,
        'Assortment_1': Assortment_1,
        'Assortment_2': Assortment_2,
        **day_of_week_encoded
    }

    return pd.DataFrame([data])

# --- Run Model ---
input_df = user_input_features()
st.markdown("---")
st.subheader("🧾 Feature Summary")
st.write(input_df)

if st.button("🔍 Predict Sales"):
    prediction = model.predict(input_df)
    st.subheader("📈 Predicted Sales")
    st.success(f"💰 Estimated Sales: **{int(prediction[0]):,} units**")
