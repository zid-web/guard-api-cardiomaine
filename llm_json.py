"""
Helpers pour parser le JSON renvoyé par les LLM (Claude, etc.).

Les modèles ajoutent parfois des fences markdown, des virgules traînantes,
des caractères de contrôle, ou tronquent la réponse (max_tokens) — ce qui
provoque `json.JSONDecodeError: Expecting value: line N column M`.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional


_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_CTRL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("```json", "").replace("```JSON", "").replace("```", "")
    return text.strip()


def extract_json_object(text: str) -> str:
    """Retourne le sous-texte entre le premier `{` et le dernier `}`."""
    text = strip_code_fences(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Aucun objet JSON trouvé dans la réponse du modèle")
    return text[start : end + 1]


def repair_common_json_issues(text: str) -> str:
    """Réparations heuristiques sans dépendance externe."""
    text = _CTRL_CHARS_RE.sub("", text)
    # Virgules traînantes avant } ou ]
    prev = None
    while prev != text:
        prev = text
        text = _TRAILING_COMMA_RE.sub(r"\1", text)
    # Valeurs manquantes : "key": , ou "key": } → null
    text = re.sub(r":\s*,", ": null,", text)
    text = re.sub(r":\s*}", ": null}", text)
    text = re.sub(r":\s*]", ": null]", text)
    # True/False/None / undefined style → JSON
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    text = re.sub(r"\bundefined\b", "null", text)
    # Guillemets typographiques
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    return text


def _close_truncated_json(text: str) -> Optional[str]:
    """
    Tente de fermer un JSON tronqué en refermant les structures ouvertes.
    Utile quand stop_reason == max_tokens.
    """
    in_string = False
    escape = False
    stack: list[str] = []

    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()

    if not stack and not in_string:
        return text

    # Couper proprement après la dernière virgule/structure complète si on est
    # au milieu d'une valeur (ex: `"raw_text": "P/` → retirer la clé incomplète).
    cut = text.rstrip()
    if in_string:
        # Fermer la chaîne ouverte
        cut += '"'
    # Retirer une virgule traînante éventuelle
    cut = cut.rstrip()
    if cut.endswith(","):
        cut = cut[:-1]

    # Recompter la pile après coupure
    in_string = False
    escape = False
    stack = []
    for ch in cut:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()

    if in_string:
        cut += '"'
    while stack:
        cut += stack.pop()
    return cut


def parse_llm_json(raw_text: str) -> Any:
    """
    Parse un JSON LLM avec plusieurs stratégies de récupération.
    Lève json.JSONDecodeError si tout échoue.
    """
    if not raw_text or not str(raw_text).strip():
        raise json.JSONDecodeError("Réponse vide", raw_text or "", 0)

    stripped = strip_code_fences(raw_text)
    candidates: list[str] = []

    def _add(c: Optional[str]) -> None:
        if c and c not in candidates:
            candidates.append(c)

    _add(stripped)
    try:
        _add(extract_json_object(raw_text))
    except ValueError:
        pass

    repaired_base = [repair_common_json_issues(c) for c in list(candidates)]
    for c in repaired_base:
        _add(c)

    # Tentative de fermeture si tronqué
    for c in list(candidates):
        closed = _close_truncated_json(c)
        if closed:
            _add(closed)
            _add(repair_common_json_issues(closed))

    last_err: Optional[json.JSONDecodeError] = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            last_err = e
            continue

    if last_err:
        raise last_err
    raise json.JSONDecodeError("JSON illisible", stripped, 0)
