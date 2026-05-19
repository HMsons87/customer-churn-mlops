from pydantic import BaseModel


# ==========================================
# REQUEST SCHEMA
# ==========================================

class CustomerInput(BaseModel):
    customerID: str = "0000-TEST"
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    numAdminTickets: int = 0
    numTechTickets: int = 0


# ==========================================
# RESPONSE SCHEMA
# ==========================================

class PredictionResponse(BaseModel):
    prediction: int
    churn_probability: float