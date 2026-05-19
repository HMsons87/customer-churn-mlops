🚀 AI-Driven Customer Churn Prediction & Retention System
Business Overview:
Customer retention is significantly more cost-effective than customer acquisition. This project is a complete, end-to-end Machine Learning pipeline designed to proactively identify customers at a high risk of churning. By translating raw telecommunications data into actionable intelligence, this system empowers customer success and retention teams to make data-driven decisions in real-time.

Core Value Proposition:

Proactive Intervention: Identifies at-risk customers before they leave, allowing for targeted retention campaigns (e.g., personalized discounts or service upgrades).

User-Friendly Dashboard: A highly interactive, zero-code frontend interface built for non-technical business users to input customer parameters and instantly view churn probabilities and financial KPIs.

Microservices Architecture: Built with scalability in mind, separating the core ML engine (FastAPI Backend) from the user interface (Streamlit Frontend) for rapid deployment and high availability.

Technical Stack:

Machine Learning: XGBoost, Scikit-learn, Pandas.

Backend/API: FastAPI, Uvicorn, Pydantic.

Frontend/UI: Streamlit, Plotly.

DevOps/MLOps: Docker, Docker Compose, Git.

2. Local Setup Guide (For Developers)
(ده الدليل اللي بيشرح لأي مبرمج إزاي ينزل المشروع ويشغله على جهازه خطوة بخطوة)

🛠️ How to Run the Project Locally
This project is containerized for seamless execution across different environments. You can run it using Docker (Recommended) or standard Python virtual environments.

Prerequisites
Git

Python 3.9+ (If running without Docker)

Docker & Docker Compose (If running with Docker)

Step 1: Clone the Repository
Bash
git clone https://github.com/YOUR_USERNAME/customer-churn-mlops.git
cd customer-churn-mlops
(Note: Replace YOUR_USERNAME with your actual GitHub username)

Step 2: Choose Your Execution Method
Option A: Using Docker (Recommended - 1 Click Setup)
Ensure Docker Desktop is running, then execute:

Bash
docker-compose up --build
The API will be available at: http://localhost:8000

The Dashboard will be available at: http://localhost:8501

Option B: Using Python Virtual Environment (Lightweight)
If you prefer not to use Docker, you can run the microservices manually using two separate terminals.

Terminal 1 (Backend API):

Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn api.main:app --reload
Terminal 2 (Frontend Dashboard):

Bash
source venv/bin/activate  # On Windows use: venv\Scripts\activate
python -m streamlit run dashboard/app.py
Access the Dashboard at: http://localhost:8501
