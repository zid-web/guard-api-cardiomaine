"""
API REST pour la generation des gardes/astreintes - Planning Cardiomaine
Point d'entree principal, a deployer (Render, Railway, Fly.io...)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from solver import GenerateWeekRequest, GenerateWeekResponse, generate_week
from voice_command import VoiceCommandRequest, VoiceCommandResponse, handle_voice_command

app = FastAPI(
    title="API Generation Gardes - Cardiomaine",
    description="Genere les astreintes/gardes hebdomadaires en respectant l'equite et les contraintes",
    version="1.1.0",
)

ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("GUARD_API_KEY", "")


def _check_api_key(x_api_key: str):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Cle API invalide ou manquante")


@app.api_route("/", methods=["GET", "HEAD"])
def health_check():
    return {"status": "ok", "service": "guard-generation-api"}


@app.post("/generate-week", response_model=GenerateWeekResponse)
def generate_week_endpoint(req: GenerateWeekRequest, x_api_key: str = ""):
    _check_api_key(x_api_key)
    try:
        return generate_week(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de generation: {str(e)}")


@app.post("/voice-command", response_model=VoiceCommandResponse)
def voice_command_endpoint(req: VoiceCommandRequest, x_api_key: str = ""):
    """
    Recoit un texte (transcrit cote navigateur via Web Speech API), l'interprete
    via Claude, applique la contrainte au planning et renvoie le planning recalcule.
    """
    _check_api_key(x_api_key)
    return handle_voice_command(req)
