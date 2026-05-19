FROM python:3.11-slim

WORKDIR /app

# هنسطب المكتبات مباشرة هنا بدل ما نعتمد على الملف
RUN pip install --no-cache-dir pandas==2.3.3 scikit-learn==1.8.0 xgboost==3.2.0 fastapi==0.136.1 uvicorn==0.46.0 pydantic==2.13.4 streamlit==1.57.0 requests==2.34.1 plotly==5.24.1

COPY . .