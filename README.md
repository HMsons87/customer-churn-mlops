# 🚀 AI-Driven Customer Churn Prediction & Retention System

Customer retention is significantly more cost-effective than customer acquisition.  
This project is a complete, end-to-end **Machine Learning pipeline** designed to proactively identify customers at a high risk of churning.  
By translating raw telecommunications data into actionable intelligence, this system empowers customer success and retention teams to make **data-driven decisions in real-time**.

---

## 💡 Core Value Proposition
- **Proactive Intervention**: Identifies at-risk customers before they leave, enabling targeted retention campaigns (e.g., personalized discounts or service upgrades).  
- **User-Friendly Dashboard**: Highly interactive, zero-code frontend interface for non-technical business users to input customer parameters and instantly view churn probabilities and financial KPIs.  
- **Microservices Architecture**: Scalable design separating the ML engine (**FastAPI Backend**) from the user interface (**Streamlit Frontend**) for rapid deployment and high availability.  

---

## 🛠️ Technical Stack
- **Machine Learning**: XGBoost, Scikit-learn, Pandas  
- **Backend/API**: FastAPI, Uvicorn, Pydantic  
- **Frontend/UI**: Streamlit, Plotly  
- **DevOps/MLOps**: Docker, Docker Compose, Git  

---

## ⚙️ Local Setup Guide (For Developers)

This project is containerized for seamless execution across different environments.  
You can run it using **Docker (Recommended)** or standard **Python virtual environments**.

### 🔑 Prerequisites
- Git  
- Python 3.9+ (If running without Docker)  
- Docker & Docker Compose (If running with Docker)  

---



### 📥 Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-mlops.git
cd customer-churn-mlops
```

---

### ⚡ Step 2: Choose Your Execution Method
##### ✅ Option A: Using Docker (Recommended - 1 Click Setup)
> Ensure Docker Desktop is running, then execute:

```bash
docker-compose up --build
```

- API available at: http://localhost:8000
- Dashboard available at: http://localhost:8501

##### 🌀 Option B: Using Python Virtual Environment (Lightweight)
> If you prefer not to use Docker, run the microservices manually using two separate terminals.

#### Terminal 1 (Backend API):
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
pip install -r requirements.txt
python -m uvicorn api.main:app --reload
```
#### Terminal 2 (Frontend Dashboard):
```bash
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
python -m streamlit run dashboard/app.py
```
- Access the Dashboard at: http://localhost:8501

---
