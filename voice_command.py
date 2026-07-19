"""
Module de traitement des commandes vocales/textuelles pour modifier le planning.

Flux :
1. Le frontend envoie le texte brut (transcrit par le navigateur via Web Speech API)
   + la date du jour + la liste des médecins connus.
2. Ce module appelle Claude (API Anthropic) pour transformer le texte en instruction
   structurée et strictement validée (JSON).
3. L'instruction est ensuite appliquée comme contrainte forcée dans le solveur
   (via `existing_schedule`), et `generate_week()` est rappelé : le solveur CP-SAT
   recalcule automatiquement tout le planning en respectant cette contrainte ET
   toutes les règles métier existantes (c'est la "cascade" demandée).
"""

import os
import json
from datetime import date, timedelta
from typing import List, Optional

from fastapi import HTTPException
from pydantic import BaseModel
import anthropic

from solver import (
    GenerateWeekRequest,
    GenerateWeekResponse,
    Medecin,
    generate_week,
    map_row_key_to_slot_activity,
    DAY_NAMES_FR,
)

client = anthropic.Anthropic()  # lit ANTHROPIC_API_KEY depuis les variables d'environnement

# ============================================================
# Modèles
# ============================================================

class VoiceCommandRequest(BaseModel):
    text: str                      # texte transcrit ("demain S remplace B en garde")
    reference_date: str            # date du jour, format YYYY-MM-DD (envoyée par le frontend)
    known_doctors: List[str]       # liste des codes médecins valides, ex: ["W","O","M","A","Z","CH","FV"]
    # Le planning complet actuel (pour reconstruire la requête de génération après modification)
    current_week_request: GenerateWeekRequest


class ParsedCommand(BaseModel):
    date: str                      # YYYY-MM-DD résolu
    slot: str                      # "matin" | "am" | "nuit"
    activity: str                  # "ASTREINTE" | "GARDE" | "CORO" | "NCT"
    doctor_out: Optional[str] = None   # médecin remplacé (None si simple ajout)
    doctor_in: str                 # médecin qui prend le créneau
    confidence: str                # "high" | "low" - si "low", le frontend doit demander confirmation


class VoiceCommandResponse(BaseModel):
    parsed_command: ParsedCommand
    updated_schedule: GenerateWeekResponse
    message: str                   # résumé lisible pour confirmation à l'utilisateur


# ============================================================
# Étape 1 : transformer le texte en instruction structurée via Claude
# ============================================================

SYSTEM_PROMPT = """Tu transformes une consigne orale ou écrite en français, concernant un planning \
médical de gardes/astreintes, en une instruction JSON strictement structurée.

Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ni après, sans balises markdown.

Format exact attendu :
{
  "date": "YYYY-MM-DD",
  "slot": "matin" | "am" | "nuit",
  "activity": "ASTREINTE" | "GARDE" | "CORO" | "NCT",
  "doctor_out": "CODE_MEDECIN ou null",
  "doctor_in": "CODE_MEDECIN",
  "confidence": "high" | "low"
}

Règles :
- Résous les expressions relatives ("demain", "après-demain", "lundi prochain") à partir de la date de référence fournie.
- Les codes médecins doivent être EXACTEMENT l'un de ceux fournis dans la liste des médecins connus. \
Si le texte mentionne un nom qui ne correspond à aucun code connu, mets "confidence": "low".
- Si l'activité ou le créneau n'est pas explicite dans le texte, déduis le plus probable \
(la garde de nuit est l'activité la plus fréquente pour ce type de consigne), mais mets "confidence": "low" \
si tu as dû deviner.
- Si la consigne ne mentionne pas de remplacement explicite (ex: "S est de garde demain" sans mention \
d'un autre médecin), mets "doctor_out": null.
- Ne réponds jamais avec autre chose qu'un JSON valide.
"""


def parse_command_with_claude(text: str, reference_date: str, known_doctors: List[str]) -> ParsedCommand:
    user_prompt = f"""Date de référence (aujourd'hui) : {reference_date}
Médecins connus (codes valides) : {", ".join(known_doctors)}

Consigne à interpréter : "{text}"
"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw_text = response.content[0].text.strip()
        # Sécurité : au cas où le modèle ajouterait des backticks malgré la consigne
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw_text)
        return ParsedCommand(**data)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise HTTPException(
            status_code=422,
            detail=f"Impossible d'interpréter la consigne vocale : {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Erreur lors de l'appel au service d'interprétation : {str(e)}"
        )


# ============================================================
# Étape 2 : appliquer l'instruction structurée au planning (cascade via le solveur)
# ============================================================

# Mapping inverse : (slot, activity) -> row_key utilisé par existing_schedule
_SLOT_ACTIVITY_TO_ROW_KEY = {
    ("matin", "ASTREINTE"): "Astreintes ATL Matin",
    ("am", "ASTREINTE"): "Astreintes ATL Midi",
    ("nuit", "ASTREINTE"): "Astreintes ATL Nuit",
    ("matin", "GARDE"): "Garde Matin",
    ("am", "GARDE"): "Garde Midi",
    ("nuit", "GARDE"): "Garde Nuit",
    ("matin", "CORO"): "Matin - Coro",
    ("am", "CORO"): "Apm - Coro",
    ("nuit", "NCT"): "Hors site - NCT",
}


def apply_command_to_schedule(
    cmd: ParsedCommand,
    current_request: GenerateWeekRequest,
) -> GenerateWeekResponse:
    """
    Force le médecin `doctor_in` sur le créneau demandé, en écrasant toute
    saisie existante à cet endroit, puis relance le solveur.
    Le solveur recalcule alors automatiquement TOUT le reste du planning
    (équité, séquences, repos, alternances) en tenant compte de cette contrainte.
    """
    row_key = _SLOT_ACTIVITY_TO_ROW_KEY.get((cmd.slot, cmd.activity))
    if row_key is None:
        raise HTTPException(
            status_code=422,
            detail=f"Combinaison créneau/activité non reconnue : {cmd.slot} / {cmd.activity}"
        )

    target_date = date.fromisoformat(cmd.date)
    day_name = DAY_NAMES_FR[target_date.weekday()]

    # Reconstruit existing_schedule à partir de celui déjà présent (préserve les autres saisies)
    existing = dict(current_request.existing_schedule or {})
    existing[f"{row_key}||{day_name}"] = [cmd.doctor_in]

    updated_request = current_request.model_copy(update={"existing_schedule": existing})

    return generate_week(updated_request)


# ============================================================
# Point d'entrée combiné (à appeler depuis l'endpoint FastAPI)
# ============================================================

def handle_voice_command(req: VoiceCommandRequest) -> VoiceCommandResponse:
    parsed = parse_command_with_claude(req.text, req.reference_date, req.known_doctors)

    if parsed.doctor_in not in req.known_doctors:
        raise HTTPException(
            status_code=422,
            detail=f"Médecin '{parsed.doctor_in}' non reconnu. Médecins valides : {req.known_doctors}"
        )

    updated_schedule = apply_command_to_schedule(parsed, req.current_week_request)

    replacement_txt = f" (remplace {parsed.doctor_out})" if parsed.doctor_out else ""
    message = (
        f"{parsed.doctor_in} affecté(e) le {parsed.date} "
        f"({parsed.slot}, {parsed.activity}){replacement_txt}. "
        f"Planning recalculé automatiquement."
    )
    if parsed.confidence == "low":
        message += " ⚠️ Confiance faible sur l'interprétation — vérifiez avant de valider."

    return VoiceCommandResponse(
        parsed_command=parsed,
        updated_schedule=updated_schedule,
        message=message,
    )
