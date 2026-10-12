"""
Consignes envoyées par l'application (bloc `app_rules`, voir
planning-cardiomaine/docs/SOLVER-CONTRACT.md).

Appliquées comme contraintes DURES sur les variables déjà créées du modèle
(x[(doc, d_idx, slot, activity)]) : elles ne créent jamais de variable et ne
peuvent donc que restreindre. Toute clé absente ou mal formée est ignorée
(compatibilité ascendante) ; le résumé renvoyé alimente `applied_rules`.
"""
from typing import Any, Dict, List, Tuple

DAY_NAMES_FR = ["LUNDI", "MARDI", "MERCREDI", "JEUDI", "VENDREDI", "SAMEDI", "DIMANCHE"]
ROW_TO_SLOT_ACTIVITY = {
    "Astreintes ATL Matin": ("matin", "ASTREINTE"),
    "Astreintes ATL Midi": ("am", "ASTREINTE"),
    "Astreintes ATL Nuit": ("nuit", "ASTREINTE"),
    "Garde Matin": ("matin", "GARDE"),
    "Garde Midi": ("am", "GARDE"),
    "Garde Nuit": ("nuit", "GARDE"),
    "Matin - Coro": ("matin", "CORO"),
    "Apm - Coro": ("am", "CORO"),
    "Hors site - NCT": ("nuit", "NCT"),
    "Apm - RÉEDUCATION": ("am", "REEDUC"),
    "Pré-op": ("am", "PRE_OP"),
    "Matin - Rythmo": ("matin", "RYTHMO"),
    "Apm - Rythmo": ("am", "RYTHMO"),
}


def _vars_for_row(x: Dict[tuple, Any], row: str, d_idx: int) -> List[Tuple[str, Any]]:
    """Variables (doc, var) correspondant à une ligne de l'appli un jour donné."""
    if row in ROW_TO_SLOT_ACTIVITY:
        slot, act = ROW_TO_SLOT_ACTIVITY[row]
        return [(k[0], v) for k, v in x.items() if k[1] == d_idx and k[2] == slot and k[3] == act]
    hist = f"HIST::{row}"
    return [(k[0], v) for k, v in x.items() if k[1] == d_idx and k[3] == hist]


def apply_app_rules(model, x: Dict[tuple, Any], app_rules: Dict[str, Any], doctor_ids: set, warnings: List[str]) -> Dict[str, Any]:
    applied: Dict[str, Any] = {"version": app_rules.get("version")}

    # 1. Cases fermées (fériés, vacations fermées, NCT sans vacation au calendrier)
    n_closed = 0
    for c in app_rules.get("closed_slots") or []:
        try:
            d_idx = DAY_NAMES_FR.index(c["day"])
            for _doc, v in _vars_for_row(x, c["row"], d_idx):
                # les gardes et astreintes ATL restent ouvertes les jours fériés (déjà exclues côté app)
                model.Add(v == 0)
                n_closed += 1
        except (KeyError, ValueError, TypeError):
            continue
    applied["closed_slots_vars_forced_off"] = n_closed

    # 2. Gardes de nuit
    ng = app_rules.get("night_guard") or {}
    night = {(k[0], k[1]): v for k, v in x.items() if k[2] == "nuit" and k[3] == "GARDE"}
    for doc in ng.get("never_doctors") or []:
        for (d, di), v in night.items():
            if d == doc:
                model.Add(v == 0)
    pool = ng.get("friday_pool")
    if pool:
        for (d, di), v in night.items():
            if di == 4 and d not in pool:
                model.Add(v == 0)
    for doc in ng.get("never_friday") or []:
        v = night.get((doc, 4))
        if v is not None:
            model.Add(v == 0)
    if ng.get("no_consecutive_nights"):
        docs = {d for (d, _di) in night}
        for doc in docs:
            for di in range(0, 6):
                a, b = night.get((doc, di)), night.get((doc, di + 1))
                if a is not None and b is not None:
                    model.Add(a + b <= 1)
        # le dimanche précédent est déjà géré par previous_sunday_guard_doctor (repos lundi)
        prev = ng.get("previous_sunday_guard_doctor")
        if prev and (prev, 0) in night:
            model.Add(night[(prev, 0)] == 0)
        applied["no_consecutive_nights"] = True

    # 3. Limites hebdomadaires
    wl = app_rules.get("weekly_limits") or {}
    stress = wl.get("stress") or {}
    if stress.get("max_per_doctor_per_week"):
        rows = set(stress.get("rows") or [])
        excluded = set(stress.get("excluded_from_limit") or [])
        by_doc: Dict[str, List[Any]] = {}
        cells = set()
        for k, v in x.items():
            if k[3].startswith("HIST::") and k[3][6:] in rows:
                cells.add((k[1], k[3]))
                by_doc.setdefault(k[0], []).append(v)
        eligible_docs = [d for d in by_doc if d not in excluded]
        if len(eligible_docs) >= len(cells) and cells:
            for d in eligible_docs:
                model.Add(sum(by_doc[d]) <= int(stress["max_per_doctor_per_week"]))
            applied["stress_once_per_week"] = True
        elif cells:
            warnings.append("Stress : pas assez de médecins disponibles pour une seule vacation chacun cette semaine — règle assouplie.")
    if (wl.get("pre_op") or {}).get("distinct_doctor_per_day"):
        by_doc = {}
        days_with = set()
        for k, v in x.items():
            if k[3] == "PRE_OP":
                by_doc.setdefault(k[0], []).append(v)
                days_with.add(k[1])
        if len(by_doc) >= len(days_with) and days_with:
            for vs in by_doc.values():
                model.Add(sum(vs) <= 1)
            applied["pre_op_distinct"] = True

    return applied
