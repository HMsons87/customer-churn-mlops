import streamlit as st
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="Churn Prediction", page_icon="📉", layout="wide")

st.title("Customer Churn Prediction Dashboard")
st.markdown("Enter the customer's data below to predict their churn risk and view insights.")
st.markdown("---")

# تقسيم الشاشة لعمودين (اليمين للنتيجة والجراف، والشمال للإدخال)
input_col, result_col = st.columns([1.5, 1])

with input_col:
    st.subheader("Customer Details")
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen (1=Yes, 0=No)", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    predict_btn = st.button("Predict Churn 🚀", width="stretch")

with result_col:
    st.subheader("📊 AI Prediction & Insights")
    
    # مكان فارغ هيتملي لما ندوس على الزرار
    result_placeholder = st.empty()

if predict_btn:
    payload = {
        "customerID": "0000-TEST",
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "numAdminTickets": 0,
        "numTechTickets": 0
    }

    try:
        response = requests.post("http://api:8000/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            probability = result.get("churn_probability", 0) * 100
            
            with result_placeholder.container():
                # 1. عرض الرسالة التحذيرية أو الآمنة
                if result.get("prediction") == 1:
                    st.error(f"⚠️ High Risk of Churn!")
                    bar_color = "red"
                else:
                    st.success(f"✅ Customer is Safe.")
                    bar_color = "green"
                
                # 2. رسم عداد الاحتمالية (Gauge Chart)
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability,
                    number = {'suffix': "%"},
                    title = {'text': "Churn Risk Probability"},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': bar_color},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': "rgba(0, 255, 0, 0.2)"},
                            {'range': [30, 70], 'color': "rgba(255, 255, 0, 0.2)"},
                            {'range': [70, 100], 'color': "rgba(255, 0, 0, 0.2)"}],
                    }
                ))
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, width="stretch")
                
                # 3. إحصائيات مالية سريعة للعميل (KPIs)
                st.markdown("#### 💰 Customer Value Overview")
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric(label="Tenure", value=f"{tenure} mos")
                kpi2.metric(label="Monthly", value=f"${monthly_charges:.2f}")
                kpi3.metric(label="Total Value", value=f"${total_charges:.2f}")

        else:
            st.error(f"API Error: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to API. Is FastAPI running on port 8000?")