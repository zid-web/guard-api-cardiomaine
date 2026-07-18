"""
API REST pour la generation des gardes/astreintes - Planning Cardiomaine
Point d'entree principal, a deployer (Render, Railway, Fly.io...)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os

from solver import GenerateWeekRequest, GenerateWeekResponse, generate_week

app = FastAPI(
    title="API Generation Gardes - Cardiomaine",
    description="Genere les astreintes/gardes hebdomadaires en respectant l'equite et les contraintes",
    version="1.0.0",
)

# Autorise les appels depuis ton appli Vercel (a restreindre a ton domaine en prod)
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Cle API simple pour proteger l'endpoint (a definir sur l'hebergeur)
API_KEY = os.environ.get("GUARD_API_KEY", "")


@app.get("/")
def health_check():
    return {"status": "ok", "service": "guard-generation-api"}


@app.post("/generate-week", response_model=GenerateWeekResponse)
def generate_week_endpoint(req: GenerateWeekRequest, x_api_key: str = ""):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Cle API invalide ou manquante")
    try:
        return generate_week(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de generation: {str(e)}")
