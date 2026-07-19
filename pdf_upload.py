"""
Extraction du planning à partir d'un PDF scanné (photo/scan d'un tableau papier).

Le PDF uploadé n'a pas de texte numérique exploitable (c'est une image scannée),
donc une extraction classique par tableau (pdfplumber) ou OCR brut (tesseract)
est trop fragile face à un tableau manuscrit, avec ratures et annotations.

Approche retenue : convertir la page PDF en image, puis utiliser Claude (vision)
pour lire et structurer le contenu du tableau en JSON, en distinguant :
- les lignes reconnues comme correspondant aux catégories gérées par le solveur
  (astreintes ATL, gardes, coro, NCT) -> directement exploitables comme
  `existing_schedule` pour forcer/préserver ces créneaux,
- les autres lignes (consultations, échographies, congés, etc.) -> non gérées
  par le solveur actuellement, renvoyées telles quelles pour affichage/contrôle.

Important : la lecture d'un scan manuscrit n'est jamais fiable à 100%.
Chaque cellule ambiguë doit être signalée pour vérification humaine avant
d'être injectée dans le solveur.
"""

import base64
import io
import json
import re
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from fastapi import HTTPException
from pydantic import BaseModel
import anthropic

from solver import map_row_key_to_slot_activity

client = anthropic.Anthropic()

# Lignes que le solveur sait gérer (doivent correspondre exactement aux clés
# utilisées dans solver.map_row_key_to_slot_activity)
KNOWN_ROW_KEYS = [
    "Astreintes ATL Matin",
    "Astreintes ATL Midi",
    "Astreintes ATL Nuit",
    "Garde Matin",
    "Garde Midi",
    "Garde Nuit",
    "Matin - Coro",
    "Apm - Coro",
    "Hors site - NCT",
]

DAY_NAMES_FR = ["LUNDI", "MARDI", "MERCREDI", "JEUDI", "VENDREDI", "SAMEDI", "DIMANCHE"]


# ============================================================
# Modèles
# ============================================================

class ExtractedCell(BaseModel):
    day_name: str                  # "LUNDI".."DIMANCHE"
    doctors: List[str]             # codes extraits, ex: ["P", "M"] pour une cellule "P/M"
    raw_text: str                  # texte brut tel que lu sur le scan
    confidence: str                # "high" | "low"


class ExtractedRow(BaseModel):
    row_label: str                 # libellé de ligne tel que lu ("astreintes ATL Nuit", "ECHO1/2", ...)
    matched_row_key: Optional[str] = None   # clé connue du solveur si reconnue, sinon None
    cells: List[ExtractedCell]


class PdfExtractionResult(BaseModel):
    week_label: Optional[str] = None       # ex: "SEMAINE 29"
    dates_by_day: Dict[str, str] = {}      # {"LUNDI": "2026-07-13", ...} si lisibles
    rows: List[ExtractedRow]
    warnings: List[str] = []


class PdfUploadResponse(BaseModel):
    raw_extraction: PdfExtractionResult
    mapped_existing_schedule: Dict[str, List[str]]  # clé sérialisée "row_key||day_name" -> [doctors]
    warnings: List[str]


# ============================================================
# Étape 1 : PDF -> image
# ============================================================

def pdf_to_image_base64(file_bytes: bytes, dpi: int = 200) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page = doc[0]
        pix = page.get_pixmap(dpi=dpi)
        png_bytes = pix.tobytes("png")
        return base64.b64encode(png_bytes).decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PDF illisible : {str(e)}")


# ============================================================
# Étape 2 : image -> extraction structurée via Claude Vision
# ============================================================

EXTRACTION_SYSTEM_PROMPT = f"""Tu lis un planning médical hebdomadaire scanné (tableau papier photographié).
Le tableau a des colonnes = jours de la semaine (LUNDI à DIMANCHE, avec leurs dates)
et des lignes = catégories d'activité (astreintes, gardes, consultations, échographies, congés, etc.).

Certaines lignes correspondent exactement à ces catégories connues (utilise ces libellés EXACTS
dans "matched_row_key" quand tu les reconnais, sinon laisse "matched_row_key": null) :
{json.dumps(KNOWN_ROW_KEYS, ensure_ascii=False)}

Réponds UNIQUEMENT avec un JSON valide, sans texte avant/après, sans balises markdown, au format :
{{
  "week_label": "SEMAINE XX ou null",
  "dates_by_day": {{"LUNDI": "YYYY-MM-DD", "MARDI": "YYYY-MM-DD", ...}},
  "rows": [
    {{
      "row_label": "libellé de ligne tel que lu sur le scan",
      "matched_row_key": "une des clés connues ci-dessus, ou null",
      "cells": [
        {{"day_name": "LUNDI", "doctors": ["W"], "raw_text": "W", "confidence": "high"}},
        {{"day_name": "MARDI", "doctors": ["P", "M"], "raw_text": "P/M", "confidence": "high"}}
      ]
    }}
  ],
  "warnings": ["liste de toute cellule illisible, rature, ou ambiguïté rencontrée, avec sa position (ligne, jour)"]
}}

Règles importantes :
- Une cellule peut contenir plusieurs médecins séparés par "/" ou entre parenthèses (ex: "N/P" -> ["N","P"], "N(z)" -> ["N","Z"]).
  Extrais chaque code médecin individuellement dans la liste "doctors".
- Si une cellule est vide, illisible, raturée, ou ambiguë, mets "confidence": "low" et
  ajoute une entrée dans "warnings" décrivant précisément le problème (ligne + jour concernés).
- N'invente jamais un code médecin que tu ne peux pas lire clairement : dans ce cas
  mets "doctors": [] et "confidence": "low", en expliquant pourquoi dans "warnings".
- Ne saute aucune ligne du tableau, même celles qui ne correspondent à aucune catégorie connue :
  inclus-les avec "matched_row_key": null.
- Les dates sont au format JJ/MM ou JJ/MM/AA sur le scan : déduis l'année complète à partir
  du contexte (numéro de semaine, dates visibles) et renvoie un format YYYY-MM-DD.
"""


def extract_planning_from_pdf(file_bytes: bytes) -> PdfExtractionResult:
    image_b64 = pdf_to_image_base64(file_bytes)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4000,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Extrait ce planning selon le format JSON demandé.",
                        },
                    ],
                }
            ],
        )
        raw_text = response.content[0].text.strip()
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw_text)
        return PdfExtractionResult(**data)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Réponse d'extraction invalide (JSON malformé) : {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Erreur lors de l'extraction du PDF : {str(e)}"
        )


# ============================================================
# Étape 3 : filtrer/mapper vers existing_schedule (format solveur)
# ============================================================

def build_existing_schedule(extraction: PdfExtractionResult) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Ne conserve que les lignes reconnues (matched_row_key non nul) et les cellules
    de confiance "high", pour construire le existing_schedule injectable au solveur.
    Les clés sont sérialisées "row_key||day_name" (les tuples ne passent pas en JSON HTTP).
    """
    mapped: Dict[str, List[str]] = {}
    warnings: List[str] = list(extraction.warnings)

    for row in extraction.rows:
        if row.matched_row_key is None:
            continue
        if row.matched_row_key not in KNOWN_ROW_KEYS:
            warnings.append(
                f"Ligne '{row.row_label}' : matched_row_key '{row.matched_row_key}' "
                f"non reconnu, ignorée."
            )
            continue

        for cell in row.cells:
            if cell.day_name not in DAY_NAMES_FR:
                continue
            if cell.confidence == "low":
                warnings.append(
                    f"{row.matched_row_key} / {cell.day_name} : lecture incertaine "
                    f"('{cell.raw_text}') - à vérifier avant application."
                )
                continue
            if not cell.doctors:
                continue
            key = f"{row.matched_row_key}||{cell.day_name}"
            mapped[key] = cell.doctors

    return mapped, warnings


def handle_pdf_upload(file_bytes: bytes) -> PdfUploadResponse:
    extraction = extract_planning_from_pdf(file_bytes)
    mapped_schedule, warnings = build_existing_schedule(extraction)
    return PdfUploadResponse(
        raw_extraction=extraction,
        mapped_existing_schedule=mapped_schedule,
        warnings=warnings,
    )
