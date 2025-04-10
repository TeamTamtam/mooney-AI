# Base image (Python)
FROM python:3.10

# COPY ./app
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port (FastAPI 기본 포트)
EXPOSE 8000
#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
# 운영 모드 - 멀티프로세스
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# uvicorn app.main:app --reload