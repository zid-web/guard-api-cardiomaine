import os
import base64
import json
from typing import Dict, List, Tuple
import anthropic
from fastapi import HTTPException

client = anthropic.Anthropic()

# Correspondance numérique → code médecin (à adapter selon vos PDF)
NUM_TO_DOCTOR = {
    "1": "Z", "2": "M", "3": "O", "4": "W",
    "5": "P", "6": "A", "7": "S", "8": "B",
    "9": "G", "10": "H", "11": "U", "12": "V",
    "13": "Val", "14": "K", "15": "R", "16": "T",
    "17": "CH", "18": "FV", "19": "D", "20": "DAAS",
}

# Toutes les lignes attendues (même ordre que dans la grille)
EXPECTED_ROWS = [
    "Astreintes ATL Matin", "Astreintes ATL Midi", "Astreintes ATL Nuit",
    "Garde Matin", "Garde Midi", "Garde Nuit",
    "Hors site - NCT", "Hors site - CDL", "Hors site - IRM",
    "Hors site - Scinti", "Hors site - LFB", "Hors site - PSSL",
    "Matin - Cs PSS", "Matin - Cs Tessée", "Matin - Stress",
    "Matin - ETT salle 1", "Matin - ETT salle 2",
    "Matin - EE1", "Matin - EE2", "Matin - Rythmo", "Matin - Coro",
    "Apm - Cs PSS", "Apm - Cs Tessée", "Apm - Stress",
    "Apm - ETT salle 1", "Apm - ETT salle 2", "Apm - RÉEDUCATION",
    "Apm - EE1", "Apm - EE2", "Apm - Rythmo", "Apm - Coro",
    "Entrées PSS", "Pré-op",
    "1/2 journée off Matin", "1/2 journée off Après-midi",
    "Vacances", "Congrès", "Congés", "Notes du jour",
]

DAYS_FR = ["LUNDI", "MARDI", "MERCREDI", "JEUDI", "VENDREDI", "SAMEDI", "DIMANCHE"]

async def extract_schedule_from_pdf(pdf_bytes: bytes) -> Dict[Tuple[str, str], List[str]]:
    """
    Utilise Claude Vision pour extraire le planning du PDF.
    Retourne un dict {(row_key, day): [doctors]}.
    """
    pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

    prompt = f"""Analyse ce tableau de planning médical extrait d'un PDF.

Voici la correspondance des identifiants numériques vers les codes médecins :
{json.dumps(NUM_TO_DOCTOR, indent=2)}

Extrais le tableau complet au format JSON suivant :
{{
  "LUNDI": {{ "Astreintes ATL Matin": ["Z"], "Astreintes ATL Midi": [], ... }},
  "MARDI": {{ ... }},
  ...
}}

Règles :
- Si une cellule est vide, retourne une liste vide [].
- Si plusieurs médecins sont dans la même cellule (ex: "2/3"), retourne ["M", "O"].
- Utilise exactement les noms d'activités de cette liste (ne crée pas de noms inventés) :
{json.dumps(EXPECTED_ROWS, indent=2)}
- Si une activité n'est pas présente dans le PDF, ne l'inclut pas.
- Ne retourne que le JSON, sans texte avant ni après.

PDF analysé (encodé en base64) :
{pdf_base64}
"""

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_base64}},
                    {"type": "text", "text": prompt}
                ]
            }],
        )
        raw_text = response.content[0].text
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw_text)

        existing_schedule = {}
        for day, activities in data.items():
            day_upper = day.upper()
            for activity, doctors_list in activities.items():
                if activity not in EXPECTED_ROWS:
                    continue
                if isinstance(doctors_list, str):
                    doctors_list = [doctors_list]
                existing_schedule[(activity, day_upper)] = doctors_list

        return existing_schedule

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'extraction par Claude : {str(e)}")
