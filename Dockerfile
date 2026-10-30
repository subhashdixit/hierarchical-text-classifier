FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PORT=8000
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY wsgi.py gunicorn.conf.py ./
COPY models ./models
COPY reports/metrics.json ./reports/metrics.json
RUN pip install --no-cache-dir --no-deps . && useradd --create-home --uid 10001 appuser && chmod -R a+rX /app
# The installed package must resolve artifact paths independently of site-packages.
ENV MODEL_DIR=/app/models APP_ROOT=/app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/health/ready')"
CMD ["gunicorn", "--config", "gunicorn.conf.py", "wsgi:app"]
