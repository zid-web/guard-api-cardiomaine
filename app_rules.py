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


def apply_app_rules(model, x: Dict[tuple, Any], app_rules: Dict[str, Any], doctor_ids: set, warnings: List[str],
                    existing_schedule: Dict[str, List[str]] = None) -> Tuple[Dict[str, Any], List[Any]]:
    """Retourne (résumé accusé de réception, termes de bonus à soustraire de l'objectif)."""
    applied: Dict[str, Any] = {"version": app_rules.get("version")}
    bonus: List[Any] = []
    existing_schedule = existing_schedule or {}
    NURSES = {"Val", "Véro", "Laura"}

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

    # 4. Cases saisies / vidées par l'admin (rows connues du solveur uniquement)
    n_fixed = n_cleared = 0
    for c in app_rules.get("cleared_cells") or []:
        try:
            d_idx = DAY_NAMES_FR.index(c["day"])
            for _doc, v in _vars_for_row(x, c["row"], d_idx):
                model.Add(v == 0)
                n_cleared += 1
        except (KeyError, ValueError, TypeError):
            continue
    for c in app_rules.get("fixed_assignments") or []:
        try:
            d_idx = DAY_NAMES_FR.index(c["day"])
            docs = set(c["doctors"])
            for doc, v in _vars_for_row(x, c["row"], d_idx):
                if doc in docs:
                    bonus.append(50 * v)  # souple : une saisie validée ne doit pas rendre le modèle infaisable
                else:
                    model.Add(v == 0)  # la case est déjà occupée
                    n_fixed += 1
        except (KeyError, ValueError, TypeError):
            continue
    applied["fixed_cells_protected"] = n_fixed
    applied["cleared_cells_respected"] = n_cleared

    # 5. Éligibilités
    el = app_rules.get("eligibility") or {}
    ett_forbidden = set(el.get("ett_forbidden") or [])
    solo = set(el.get("ee_solo_doctors") or [])
    for k, v in x.items():
        act = k[3]
        if not act.startswith("HIST::"):
            continue
        row = act[6:]
        if "ETT" in row and k[0] in ett_forbidden:
            model.Add(v == 0)
        if el.get("ee_solo_doctors") is not None and "EE" in row and row.endswith(("EE1", "EE2")):
            # sans infirmière dans la case : seuls O, V, DAAS peuvent occuper EE
            has_nurse = any(n in NURSES for n in existing_schedule.get(f"{row}||{DAY_NAMES_FR[k[1]]}", []))
            if not has_nurse and k[0] not in solo:
                model.Add(v == 0)
    if el.get("ee_solo_doctors") is not None:
        # même médecin sur EE1 et EE2 quand les deux cases ont la même infirmière
        for d_idx, day in enumerate(DAY_NAMES_FR[:5]):
            for part in ("Matin", "Apm"):
                r1, r2 = f"{part} - EE1", f"{part} - EE2"
                n1 = {n for n in existing_schedule.get(f"{r1}||{day}", []) if n in NURSES}
                n2 = {n for n in existing_schedule.get(f"{r2}||{day}", []) if n in NURSES}
                if n1 and n1 == n2:
                    v1 = {d: v for d, v in _vars_for_row(x, r1, d_idx)}
                    v2 = {d: v for d, v in _vars_for_row(x, r2, d_idx)}
                    for d in set(v1) | set(v2):
                        if d in v1 and d in v2:
                            model.Add(v1[d] == v2[d])
                        else:
                            model.Add((v1.get(d) or v2.get(d)) == 0)
        applied["ee_rules"] = True

    # 6. Jours habituels des vacations hors site
    usual = app_rules.get("hors_site_usual_days") or {}
    for k, v in x.items():
        if k[3].startswith("HORSSITE::"):
            row = k[3][len("HORSSITE::"):]
            if row in usual and DAY_NAMES_FR[k[1]] not in usual[row]:
                model.Add(v == 0)
    applied["hors_site_usual_days"] = bool(usual)

    # 7. Priorités (bonus souples dans l'objectif)
    pr = app_rules.get("priorities") or {}
    coro_first = set(pr.get("coro_first") or [])
    echo = set(pr.get("echo_group") or [])
    for k, v in x.items():
        act = k[3]
        if act == "CORO" and k[0] in coro_first:
            bonus.append(3 * v)
        elif act.startswith("HIST::") and echo:
            row = act[6:]
            is_echo_row = "ETT" in row or row.endswith("Stress")
            is_cs = " - Cs " in row
            if is_echo_row and k[0] in echo:
                bonus.append(4 * v)
            elif is_cs and k[0] not in echo:
                bonus.append(2 * v)
    applied["priorities"] = bool(pr)

    return applied, bonus
