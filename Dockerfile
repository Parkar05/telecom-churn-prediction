# Container image for running the app on Azure App Service / Container Apps
# or any internal server. Build: docker build -t churn-app .
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY artifacts/ artifacts/
COPY app.py .
COPY assets/ assets/
COPY docs/architecture.png docs/architecture.png
COPY .streamlit/config.toml .streamlit/config.toml

# Run as a non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
