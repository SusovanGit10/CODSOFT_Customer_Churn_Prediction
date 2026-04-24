import streamlit as st
import joblib
import pandas as pd

# Load model & preprocessors
model = joblib.load("../models/model.pkl")
scaler = joblib.load("../models/scaler.pkl")
le_gender = joblib.load("../models/le_gender.pkl")
le_geo = joblib.load("../models/le_geo.pkl")

FEATURE_COLUMNS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
.block-container { padding-top: 2rem; }
.main-title { font-size: 38px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="main-title">📊 Customer Churn Prediction</div>', unsafe_allow_html=True)
st.write("Predict churn risk and understand key drivers using machine learning")
st.markdown("---")

# ---------- SESSION STATE ----------
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "prob" not in st.session_state:
    st.session_state.prob = None

# ---------- SIDEBAR ----------
st.sidebar.header("🧾 Enter Customer Details")

credit_score = st.sidebar.number_input("Credit Score", 300, 900, 600)
geography = st.sidebar.selectbox("Geography", le_geo.classes_)
gender = st.sidebar.selectbox("Gender", le_gender.classes_)
age = st.sidebar.slider("Age", 18, 100, 30)
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 3)

balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 50000.0)
products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
credit_card = st.sidebar.selectbox("Has Credit Card", ["Yes", "No"])
active = st.sidebar.selectbox("Is Active Member", ["Yes", "No"])
salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

# Reset button
if st.sidebar.button("🔄 Reset"):
    st.session_state.prediction_done = False

# ---------- MAIN LAYOUT ----------
col1, col2 = st.columns([2, 1])

# ---------- LEFT: SUMMARY ----------
with col1:
    st.subheader("📋 Customer Summary")

    c1, c2 = st.columns(2)

    with c1:
        st.info(f"""
        **Credit Score:** {credit_score}  
        **Age:** {age}  
        **Tenure:** {tenure} years  
        """)

    with c2:
        st.info(f"""
        **Balance:** ₹{balance:,.0f}  
        **Products:** {products}  
        **Active Member:** {active}  
        """)

# ---------- RIGHT: PREDICTION ----------
with col2:
    st.subheader("🔍 Prediction")

    if st.button("Predict", use_container_width=True):

        with st.spinner("Analyzing customer data..."):

            data = pd.DataFrame([{
                "CreditScore": credit_score,
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": products,
                "HasCrCard": 1 if credit_card == "Yes" else 0,
                "IsActiveMember": 1 if active == "Yes" else 0,
                "EstimatedSalary": salary
            }])

            # Encode
            data["Geography"] = le_geo.transform(data["Geography"])
            data["Gender"] = le_gender.transform(data["Gender"])

            # Order
            data = data[FEATURE_COLUMNS]

            # Scale
            data_scaled = scaler.transform(data)

            # Predict
            prediction = model.predict(data_scaled)[0]
            prob = model.predict_proba(data_scaled)[0][1]

        # Save state
        st.session_state.prediction_done = True
        st.session_state.prediction = prediction
        st.session_state.prob = prob

    # ---------- RESULT ----------
    if st.session_state.prediction_done:

        prediction = st.session_state.prediction
        prob = st.session_state.prob

        st.markdown("---")
        st.subheader("📈 Result")

        # Result card
        if prediction == 1:
            st.error("🚨 High Risk — Immediate retention action recommended")
        else:
            st.success("✅ Low Risk — Customer likely to stay")

        # Probability
        st.metric("Churn Probability", f"{prob:.2%}")

        st.progress(int(prob * 100))

        # Risk messaging
        if prob < 0.3:
            st.success("🟢 Low Risk — No action needed")
        elif prob < 0.7:
            st.warning("🟡 Medium Risk — Consider engagement strategies")
        else:
            st.error("🔴 High Risk — Take retention action")

        # ---------- CHART ----------
        st.subheader("📊 Risk Distribution")

        chart_data = pd.DataFrame({
            "Category": ["Safe", "Churn"],
            "Probability": [1 - prob, prob]
        })

        st.bar_chart(chart_data.set_index("Category"))

        # ---------- FEATURE IMPORTANCE ----------
        if hasattr(model, "feature_importances_"):
            st.subheader("📌 Feature Importance")

            importance_df = pd.DataFrame({
                "Feature": FEATURE_COLUMNS,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            st.bar_chart(importance_df.set_index("Feature"))

            # ---------- KEY DRIVERS ----------
            st.subheader("🧠 Key Drivers")

            top_features = importance_df.head(3)

            for _, row in top_features.iterrows():
                st.write(f"• {row['Feature']} strongly influences prediction")

    else:
        st.info("👈 Enter details and click Predict")

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Built with Machine Learning • Streamlit • Customer Churn Prediction System")