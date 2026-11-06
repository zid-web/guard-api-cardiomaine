FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000
# Journaux affichés immédiatement dans Render (sinon les print(), dont l'avertissement
# « GUARD_API_KEY n'est pas definie », restent bloqués dans le tampon de sortie).
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
