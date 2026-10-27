FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 BI_DATA_DIR=/app/data/generated

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Данные генерируются при сборке образа, чтобы первый запуск был мгновенным
RUN python -m bi.synth /app/data/generated

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
