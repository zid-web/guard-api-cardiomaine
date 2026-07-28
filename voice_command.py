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
from pydantic import BaseModel, ValidationError
import anthropic

from llm_json import parse_llm_json

from solver import (
    GenerateWeekRequest,
    GenerateWeekResponse,
    Medecin,
    RoomMaintenance,
    PartialAbsence,
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
    command_type: str = "assignment"  # "assignment" | "room_maintenance" | "partial_absence"
    date: str                      # YYYY-MM-DD résolu (date de début pour room_maintenance)
    slot: str                      # "matin" | "am" | "nuit" (ignoré pour room_maintenance)
    activity: str                  # "ASTREINTE" | "GARDE" | "CORO" | "NCT" (ignoré hors "assignment")
    doctor_out: Optional[str] = None   # médecin remplacé (None si simple ajout)
    # Optional à la validation brute : Claude renvoie parfois null ; normalisé ensuite.
    # Obligatoire uniquement pour command_type="assignment" (voir _parse_command_items).
    doctor_in: Optional[str] = None
    confidence: str = "low"        # "high" | "low" - si "low", le frontend doit demander confirmation
    # room_maintenance uniquement : fin de la période (par défaut = date si absent, 1 seul jour)
    end_date: Optional[str] = None
    # partial_absence ET room_maintenance : sous-ensemble de ["matin", "am"]
    # (room_maintenance ne concerne jamais "nuit", pas d'activité CORO la nuit)
    slots: Optional[List[str]] = None


class VoiceCommandResponse(BaseModel):
    parsed_command: ParsedCommand
    updated_schedule: GenerateWeekResponse
    message: str                   # résumé lisible pour confirmation à l'utilisateur


# ============================================================
# Étape 1 : transformer le texte en instruction structurée via Claude
# ============================================================

SYSTEM_PROMPT = """Tu transformes une consigne orale ou écrite en français, concernant un planning \
médical de gardes/astreintes/vacations, en une instruction JSON strictement structurée.

Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ni après, sans balises markdown.

Il existe TROIS types de consigne possibles, distingués par "command_type" :

**1. "assignment" (le plus fréquent - affectation/remplacement d'un médecin)**
{
  "command_type": "assignment",
  "date": "YYYY-MM-DD",
  "slot": "matin" | "am" | "nuit" | "weekend",
  "activity": "ASTREINTE" | "GARDE" | "CORO" | "NCT" | "VACANCES" | "CONGE" | "CONGRES" | "RYTHMO" | "PRE_OP" | "REEDUC",
  "doctor_out": "CODE_MEDECIN ou null",
  "doctor_in": "CODE_MEDECIN",
  "confidence": "high" | "low"
}

**2. "room_maintenance" (la salle de coronarographie est indisponible - PAS un médecin absent)**
Déclenché par des expressions comme "la salle de coro est en maintenance", "coro indisponible", \
"pas de coronarographie possible" pour une période donnée. Concerne UNIQUEMENT les créneaux \
"matin" et/ou "am" de l'activité CORO (jamais "nuit") - et UNIQUEMENT le(s) créneau(x) \
explicitement concerné(s), pas systématiquement les deux (ex: "coro après-midi indisponible" \
ne bloque PAS le matin).
{
  "command_type": "room_maintenance",
  "date": "YYYY-MM-DD",           // premier jour d'indisponibilité
  "end_date": "YYYY-MM-DD",       // dernier jour (= date si un seul jour)
  "slots": ["am"] | ["matin"] | ["matin", "am"],  // créneau(x) réellement concerné(s), déduit du texte
  "slot": "matin",                 // dupliqué ici pour cohérence de schéma (premier élément de slots)
  "activity": "CORO",              // toujours "CORO" pour ce type
  "doctor_in": null,               // PAS de médecin concerné - ne pas deviner de code médecin
  "confidence": "high" | "low"
}

Résolution des références par NUMÉRO DE SEMAINE (fréquent pour la maintenance de salle, \
ex: "de S31 à S34 inclus") : les semaines sont des semaines ISO (lundi à dimanche). Calcule \
la date du lundi de la semaine N à partir de l'année de la date de référence fournie, et pose \
"date" = lundi de la première semaine citée, "end_date" = dimanche de la dernière semaine citée. \
Si une seule semaine est citée (ex: "S31"), end_date = dimanche de cette même semaine.

**3. "partial_absence" (absence PONCTUELLE d'un médecin sur un ou des créneaux précis d'UN SEUL jour \
- différent d'un congé/vacances qui bloque la journée entière)**
Déclenché par des expressions comme "S est absent demain matin seulement", "A indisponible jeudi après-midi".
{
  "command_type": "partial_absence",
  "date": "YYYY-MM-DD",             // un seul jour, jamais une plage
  "slots": ["matin"] | ["am"] | ["nuit"] | ["matin", "am"],  // créneau(x) concerné(s), déduit du texte
  "slot": "matin",                   // dupliqué ici pour cohérence de schéma (premier élément de slots)
  "activity": "ABSENCE",             // valeur fixe pour ce type
  "doctor_in": "CODE_MEDECIN",       // le médecin absent (ici doctor_in désigne bien qui est concerné)
  "confidence": "high" | "low"
}

Règles générales :
- Résous les expressions relatives ("demain", "après-demain", "lundi prochain") à partir de la date de référence fournie.
- Les codes médecins doivent être EXACTEMENT l'un de ceux fournis dans la liste des médecins connus. \
Si le texte mentionne un nom qui ne correspond à aucun code connu, mets "confidence": "low".
- Remplacement / changement de vacation : "X remplace Y en …" → doctor_in=X, doctor_out=Y, activity selon le créneau \
(CORO, GARDE, ASTREINTE, RYTHMO, etc.) → command_type "assignment".
- Congés / vacances / absence DE PLUSIEURS JOURS ou JOURNÉE ENTIÈRE : activity "VACANCES" ou "CONGE", \
slot "matin" (journée), doctor_in = médecin absent, command_type "assignment". Si c'est PONCTUEL et sur \
un/des créneau(x) précis d'un seul jour (ex: "juste demain matin"), utilise plutôt command_type "partial_absence".
- Congrès : activity "CONGRES", command_type "assignment".
- Maintenance/indisponibilité de salle de coro (PAS un médecin) : command_type "room_maintenance", \
ne jamais mettre de code médecin dans doctor_in pour ce type.
- Si l'activité ou le créneau n'est pas explicite dans le texte, déduis le plus probable \
(la garde de nuit est l'activité la plus fréquente pour ce type de consigne), mais mets "confidence": "low" \
si tu as dû deviner.
- NCT (hors site) : utilise TOUJOURS "activity": "NCT" et "slot": "nuit" (même si le texte dit matin). \
NCT tombe en pratique le jeudi.
- Si la consigne ne mentionne pas de remplacement explicite (ex: "S est de garde demain" sans mention \
d'un autre médecin), mets "doctor_out": null.
- Pour command_type "assignment" : "doctor_in" est OBLIGATOIRE et doit toujours être une chaîne (code médecin). \
Ne renvoie JAMAIS null ni omis pour "doctor_in" dans ce cas. Pour un retrait sans remplaçant, mets doctor_in \
au médecin concerné avec activity VACANCES/CONGE, ou reformule — jamais doctor_in=null pour une "assignment".
- Pour command_type "room_maintenance" : doctor_in DOIT être null (aucun médecin concerné).
- Si tu hésites entre deux codes, choisis le plus probable et mets "confidence": "low".
- Si le texte liste PLUSIEURS dates NCT (ex: "2026-09-10 → M"), réponds avec un objet :
  { "commands": [ {…}, {…} ] } où chaque élément suit le format ci-dessus (activity NCT, slot nuit).
- Sinon réponds avec un seul objet (pas de clé "commands").
- Ne réponds jamais avec autre chose qu'un JSON valide.
"""



def _norm_doctor_code(value) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if not value or value.lower() in ("null", "none", "nil"):
        return None
    return value


def normalize_raw_command(item: dict, known_doctors: Optional[List[str]] = None) -> dict:
    """Corrige les JSON Claude invalides avant validation Pydantic.

    Cas fréquent : doctor_in=null alors que le médecin est dans doctor_out,
    ou doctor_in omis pour une absence / affectation simple.
    """
    if not isinstance(item, dict):
        raise TypeError(f"Commande attendue comme objet JSON, reçu: {type(item).__name__}")
    out = dict(item)
    command_type = out.get("command_type") or "assignment"
    out["command_type"] = command_type

    if command_type == "room_maintenance":
        # Aucun médecin concerné - ne pas essayer de deviner/récupérer doctor_in.
        out["doctor_in"] = None
        out["doctor_out"] = None
        out.setdefault("end_date", out.get("date"))
        slots = out.get("slots") or (["matin", "am"] if not out.get("slot") else [out["slot"]])
        out["slots"] = slots
        out.setdefault("slot", slots[0] if slots else "matin")
        out.setdefault("activity", "CORO")
        if not out.get("confidence"):
            out["confidence"] = "low"
        return out

    din = _norm_doctor_code(out.get("doctor_in"))
    dout = _norm_doctor_code(out.get("doctor_out"))

    if din is None and dout is not None:
        # Claude a souvent inversé / mis le seul médecin dans doctor_out
        din = dout
        dout = None

    # Alignement case / codes connus (Val vs VAL, etc.)
    known = known_doctors or []
    known_map = {k.upper(): k for k in known}

    def match_known(code: Optional[str]) -> Optional[str]:
        if code is None:
            return None
        if code in known:
            return code
        return known_map.get(code.upper(), code)

    out["doctor_in"] = match_known(din)
    out["doctor_out"] = match_known(dout)

    if command_type == "partial_absence":
        slots = out.get("slots") or ([out["slot"]] if out.get("slot") else ["matin"])
        out["slots"] = slots
        out.setdefault("slot", slots[0] if slots else "matin")
        out.setdefault("activity", "ABSENCE")

    if not out.get("confidence"):
        out["confidence"] = "low"
    return out


def _parse_command_items(data, known_doctors: List[str]) -> List[ParsedCommand]:
    if isinstance(data, dict) and isinstance(data.get("commands"), list):
        raw_items = data["commands"]
    elif isinstance(data, list):
        raw_items = data
    else:
        raw_items = [data]
    cmds: List[ParsedCommand] = []
    for item in raw_items:
        normalized = normalize_raw_command(item, known_doctors)
        cmds.append(ParsedCommand(**normalized))
    if not cmds:
        raise ValueError("Aucune commande dans la réponse LLM")
    missing = [
        i for i, c in enumerate(cmds)
        if c.command_type in ("assignment", "partial_absence") and not c.doctor_in
    ]
    if missing:
        raise ValueError(
            "Médecin destinataire (doctor_in) manquant après interprétation. "
            "Reformulez en précisant le code médecin (ex. « S est de garde mardi »)."
        )
    return cmds


def parse_commands_with_claude(text: str, reference_date: str, known_doctors: List[str]) -> List[ParsedCommand]:
    user_prompt = f"""Date de référence (aujourd'hui) : {reference_date}
Médecins connus (codes valides) : {", ".join(known_doctors)}

Consigne à interpréter : "{text}"
"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw_text = response.content[0].text.strip()
        data = parse_llm_json(raw_text)
        cmds = _parse_command_items(data, known_doctors)
        # Normalise NCT → slot nuit
        normalized: List[ParsedCommand] = []
        for cmd in cmds:
            if (cmd.activity or "").upper() == "NCT" and (cmd.slot or "").lower() != "nuit":
                cmd = cmd.model_copy(update={"slot": "nuit"})
            normalized.append(cmd)
        return normalized
    except (json.JSONDecodeError, KeyError, ValueError, TypeError, ValidationError) as e:
        raise HTTPException(
            status_code=422,
            detail=f"Impossible d'interpréter la consigne vocale : {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Erreur lors de l'appel au service d'interprétation : {str(e)}"
        )


def parse_command_with_claude(text: str, reference_date: str, known_doctors: List[str]) -> ParsedCommand:
    return parse_commands_with_claude(text, reference_date, known_doctors)[0]


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
    # Claude renvoie souvent slot=matin pour NCT → accepter tous les créneaux
    ("matin", "NCT"): "Hors site - NCT",
    ("am", "NCT"): "Hors site - NCT",
    ("weekend", "NCT"): "Hors site - NCT",
    ("matin", "VACANCES"): "Congés",
    ("am", "VACANCES"): "Congés",
    ("nuit", "VACANCES"): "Congés",
    ("weekend", "VACANCES"): "Congés",
    ("matin", "CONGE"): "Congés",
    ("am", "CONGE"): "Congés",
    ("nuit", "CONGE"): "Congés",
    ("matin", "CONGRES"): "Congrès",
    ("am", "CONGRES"): "Congrès",
    ("matin", "RYTHMO"): "Matin - Rythmo",
    ("am", "RYTHMO"): "Apm - Rythmo",
    ("matin", "PRE_OP"): "Pré-op",
    ("am", "PRE_OP"): "Pré-op",
    ("am", "REEDUC"): "Apm - RÉEDUCATION",
}


def resolve_row_key(slot: str, activity: str) -> Optional[str]:
    act = (activity or "").upper().strip()
    sl = (slot or "").lower().strip()
    if act == "NCT":
        return "Hors site - NCT"
    if act in ("VACANCES", "CONGE", "CONGES"):
        return "Congés"
    if act == "CONGRES":
        return "Congrès"
    return _SLOT_ACTIVITY_TO_ROW_KEY.get((sl, act))


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
    row_key = resolve_row_key(cmd.slot, cmd.activity)
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
    commands = parse_commands_with_claude(req.text, req.reference_date, req.known_doctors)
    if not commands:
        raise HTTPException(status_code=422, detail="Aucune consigne interprétable")

    for parsed in commands:
        if parsed.command_type == "room_maintenance":
            continue  # aucun médecin concerné, rien à valider ici
        if not parsed.doctor_in:
            raise HTTPException(
                status_code=422,
                detail="Médecin destinataire (doctor_in) manquant. Reformulez en précisant le code médecin.",
            )
        if parsed.doctor_in not in req.known_doctors:
            raise HTTPException(
                status_code=422,
                detail=f"Médecin '{parsed.doctor_in}' non reconnu. Médecins valides : {req.known_doctors}"
            )

    # Applique toutes les contraintes (affectations, maintenance salle, absences
    # ponctuelles), puis un seul generate_week (utile si plusieurs consignes
    # tombent dans la semaine courante).
    request = req.current_week_request
    existing = dict(request.existing_schedule or {})
    room_maintenance = list(request.room_maintenance or [])
    partial_absences = list(request.partial_absences or [])

    for parsed in commands:
        if parsed.command_type == "room_maintenance":
            room_maintenance.append(RoomMaintenance(
                start_date=parsed.date,
                end_date=parsed.end_date or parsed.date,
                slots=parsed.slots or ["matin", "am"],
                reason="Consigne vocale",
            ))
            continue

        if parsed.command_type == "partial_absence":
            partial_absences.append(PartialAbsence(
                doctor_id=parsed.doctor_in,
                date=parsed.date,
                slots=parsed.slots or [parsed.slot],
            ))
            continue

        # command_type == "assignment" (comportement historique inchangé)
        row_key = resolve_row_key(parsed.slot, parsed.activity)
        if row_key is None:
            raise HTTPException(
                status_code=422,
                detail=f"Combinaison créneau/activité non reconnue : {parsed.slot} / {parsed.activity}"
            )
        target_date = date.fromisoformat(parsed.date)
        day_name = DAY_NAMES_FR[target_date.weekday()]
        existing[f"{row_key}||{day_name}"] = [parsed.doctor_in]

    updated_request = request.model_copy(update={
        "existing_schedule": existing,
        "room_maintenance": room_maintenance,
        "partial_absences": partial_absences,
    })
    updated_schedule = generate_week(updated_request)
    parsed = commands[0]

    if len(commands) == 1:
        if parsed.command_type == "room_maintenance":
            slots_txt = " et ".join(parsed.slots or ["matin", "am"])
            message = (
                f"Salle de coronarographie indisponible ({slots_txt}) du {parsed.date} au "
                f"{parsed.end_date or parsed.date}. Planning recalculé automatiquement."
            )
        elif parsed.command_type == "partial_absence":
            message = (
                f"{parsed.doctor_in} absent(e) le {parsed.date} ({', '.join(parsed.slots or [parsed.slot])}). "
                f"Planning recalculé automatiquement."
            )
        else:
            replacement_txt = f" (remplace {parsed.doctor_out})" if parsed.doctor_out else ""
            message = (
                f"{parsed.doctor_in} affecté(e) le {parsed.date} "
                f"({parsed.slot}, {parsed.activity}){replacement_txt}. "
                f"Planning recalculé automatiquement."
            )
    else:
        message = (
            f"{len(commands)} contraintes appliquées (dont {parsed.date}). "
            f"Planning de la semaine courante recalculé."
        )
    if any(c.confidence == "low" for c in commands):
        message += " ⚠️ Confiance faible sur l'interprétation — vérifiez avant de valider."

    return VoiceCommandResponse(
        parsed_command=parsed,
        updated_schedule=updated_schedule,
        message=message,
    )
