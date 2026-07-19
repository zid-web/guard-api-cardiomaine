"""
Solveur complet pour le planning Cardiomaine - Version avec alternance CH / WOM
et préservation des saisies manuelles.
"""

from ortools.sat.python import cp_model
from datetime import date, timedelta
from typing import List, Dict, Optional, Literal, Set, Tuple, Any
from pydantic import BaseModel
import enum

# ============================================================
# 1. MODÈLES DE DONNÉES (entrée / sortie)
# ============================================================

class StatutMedecin(str, enum.Enum):
    PERMANENT = "permanent"
    ASTREINTE_CORO = "astreinte_coro"   # M, O, W
    FV = "fv"
    DAAS = "daas"
    D = "d"
    CH = "ch"
    ADMIN = "admin"

class Medecin(BaseModel):
    id: str
    statut: StatutMedecin
    points_astreinte: int = 0
    points_garde: int = 0
    points_nct: int = 0
    points_weekend: int = 0

class Vacation(BaseModel):
    doctor_id: str
    start_date: str
    end_date: str

class GenerateWeekRequest(BaseModel):
    week_start_date: str              # YYYY-MM-DD (lundi)
    week_type: int                    # 1 = impaire, 2 = paire
    medecins: List[Medecin]
    vacations: List[Vacation] = []
    weekend_mode: Literal["CH", "ROTATION"] = "ROTATION"
    last_nct_doctor: Optional[str] = None  # W ou M
    existing_schedule: Optional[Dict[Tuple[str, str], List[str]]] = None  # (row_key, day) -> [doctors]

class Assignment(BaseModel):
    date: str
    day_name: str
    slot: str          # "matin", "am", "nuit", "weekend"
    activity: str      # "ASTREINTE", "GARDE", "NCT", "CORO"
    doctor: str
    note: Optional[str] = None

class GenerateWeekResponse(BaseModel):
    week_start_date: str
    assignments: List[Assignment]
    warnings: List[str] = []

# ============================================================
# 2. UTILITAIRES
# ============================================================

DAY_NAMES_FR = ["LUNDI", "MARDI", "MERCREDI", "JEUDI", "VENDREDI", "SAMEDI", "DIMANCHE"]
SLOTS = ["matin", "am", "nuit"]
ACTIVITIES = ["ASTREINTE", "GARDE", "CORO", "NCT"]

# Séquences autorisées pour M/O/W (matin, am, nuit)
ALLOWED_SEQUENCES = [
    (0, 0, 0),
    (1, 1, 1),
    (2, 2, 2),
    (0, 1, 1),
    (0, 2, 2),
    (1, 1, 0),
    (2, 2, 0),
    (1, 0, 1),
    (2, 0, 2),
    (0, 1, 0),
    (0, 2, 0),
    (1, 0, 0),
    (2, 0, 0),
    (0, 0, 1),
    (0, 0, 2),
]

def is_on_vacation(doctor_id: str, day: date, vacations: List[Vacation]) -> bool:
    for v in vacations:
        if v.doctor_id == doctor_id:
            start = date.fromisoformat(v.start_date)
            end = date.fromisoformat(v.end_date)
            if start <= day <= end:
                return True
    return False

def jours_semaine(week_start: date) -> List[date]:
    return [week_start + timedelta(days=i) for i in range(7)]

def map_row_key_to_slot_activity(row_key: str) -> Tuple[Optional[str], Optional[str]]:
    """Mapper row_key vers (slot, activity)."""
    mapping = {
        "Astreintes ATL Matin": ("matin", "ASTREINTE"),
        "Astreintes ATL Midi": ("am", "ASTREINTE"),
        "Astreintes ATL Nuit": ("nuit", "ASTREINTE"),
        "Garde Matin": ("matin", "GARDE"),
        "Garde Midi": ("am", "GARDE"),
        "Garde Nuit": ("nuit", "GARDE"),
        "Matin - Coro": ("matin", "CORO"),
        "Apm - Coro": ("am", "CORO"),
        "Hors site - NCT": ("nuit", "NCT"),
    }
    return mapping.get(row_key, (None, None))

# ============================================================
# 3. SOLVEUR PRINCIPAL
# ============================================================

def generate_week(req: GenerateWeekRequest) -> GenerateWeekResponse:
    warnings = []
    week_start = date.fromisoformat(req.week_start_date)
    days = jours_semaine(week_start)

    # --- 1. Préparation des données ---
    medecins_map = {m.id: m for m in req.medecins}
    astreinte_coro_ids = {m.id for m in req.medecins if m.statut == StatutMedecin.ASTREINTE_CORO}  # W, O, M
    wom_pool = ["W", "O", "M"]  # explicit
    nct_pool = {m.id for m in req.medecins if m.statut == StatutMedecin.ASTREINTE_CORO and m.id != "O"}
    fv_id = next((m.id for m in req.medecins if m.statut == StatutMedecin.FV), None)
    daas_id = next((m.id for m in req.medecins if m.statut == StatutMedecin.DAAS), None)
    d_id = next((m.id for m in req.medecins if m.statut == StatutMedecin.D), None)

    # Demi-journées libres
    half_days_off = {
        ("MERCREDI", "am"): {"M", "W", "G", "Z", "H"},
        ("JEUDI", "am"): {"U", "S", "P"},
        ("VENDREDI", "am"): {"O", "A", "K", "R", "T"},
    }

    fixed_exclusions = {
        "P": {1},
        "U": {2},
        "A": {0, 3},
        "S": {0, 4},
    }

    # --- 2. Création des variables ---
    model = cp_model.CpModel()
    x = {}  # (doc, day_idx, slot, activity) -> BoolVar

    def add_var_if_allowed(doc_id: str, d_idx: int, slot: str, activity: str):
        day = days[d_idx]
        if is_on_vacation(doc_id, day, req.vacations):
            return

        if doc_id in (daas_id, d_id):
            return

        if doc_id == fv_id:
            if not (d_idx == 0 and slot == "nuit" and activity == "GARDE") and \
               not (d_idx == 3 and slot == "am" and activity == "CORO"):
                return

        day_name = DAY_NAMES_FR[d_idx]
        if day_name in half_days_off and slot in half_days_off[day_name]:
            if doc_id in half_days_off[day_name][slot]:
                return

        if doc_id in fixed_exclusions and d_idx in fixed_exclusions[doc_id]:
            return

        statut = medecins_map[doc_id].statut
        if statut == StatutMedecin.CH:
            return
        if statut == StatutMedecin.PERMANENT:
            if activity not in ("ASTREINTE", "GARDE"):
                return
        if statut == StatutMedecin.ASTREINTE_CORO:
            if activity == "NCT" and doc_id not in nct_pool:
                return
            if activity == "CORO" and slot not in ("matin", "am"):
                return

        var = model.NewBoolVar(f"x_{doc_id}_{d_idx}_{slot}_{activity}")
        x[(doc_id, d_idx, slot, activity)] = var

    for doc_id in medecins_map:
        for d_idx in range(7):
            for slot in SLOTS:
                for activity in ACTIVITIES:
                    add_var_if_allowed(doc_id, d_idx, slot, activity)

    # --- 3. Contraintes générales ---
    # Capacité max 2 par case
    for d_idx in range(7):
        for slot in SLOTS:
            case_vars = [v for (d, sl, act), v in x.items() if d == d_idx and sl == slot]
            if case_vars:
                model.Add(sum(case_vars) <= 2)

    # Un médecin ne fait qu'une activité par créneau
    for doc_id in medecins_map:
        for d_idx in range(7):
            for slot in SLOTS:
                slot_vars = [v for (d, sl, act), v in x.items() if d == d_idx and sl == slot and doc_id == d[0]]
                if slot_vars:
                    model.Add(sum(slot_vars) <= 1)

    # --- 4. Structure des astreintes de nuit (Lundi à Vendredi) avec alternance CH/WOM ---
    # week_type: 1 = impaire, 2 = paire
    if req.week_type == 1:
        structure = {0: "CH", 1: "CH", 2: "WOM", 3: "WOM", 4: "CH"}
    else:
        structure = {0: "WOM", 1: "WOM", 2: "CH", 3: "CH", 4: "WOM"}

    for d_idx in range(5):
        if structure[d_idx] == "CH":
            # Forcer CH sur cette nuit
            for (doc, d, sl, act), var in x.items():
                if d == d_idx and sl == "nuit" and act == "ASTREINTE":
                    if doc == "CH":
                        model.Add(var == 1)
                    else:
                        model.Add(var == 0)
        else:  # "WOM"
            wom_vars = []
            for (doc, d, sl, act), var in x.items():
                if d == d_idx and sl == "nuit" and act == "ASTREINTE":
                    if doc in wom_pool:
                        wom_vars.append(var)
                    else:
                        model.Add(var == 0)
            if wom_vars:
                model.Add(sum(wom_vars) == 1)  # exactement un médecin WOM
            else:
                warnings.append(f"Jour {DAY_NAMES_FR[d_idx]} : aucun médecin W/O/M disponible, CH utilisé")
                # Fallback CH
                for (doc, d, sl, act), var in x.items():
                    if d == d_idx and sl == "nuit" and act == "ASTREINTE":
                        if doc == "CH":
                            model.Add(var == 1)
                        else:
                            model.Add(var == 0)

    # --- 5. NCT (jeudi nuit) ---
    nct_vars = [v for (doc, d, sl, act), v in x.items() if d == 3 and sl == "nuit" and act == "NCT"]
    if nct_vars:
        model.Add(sum(nct_vars) == 1)
    else:
        warnings.append("JEUDI : aucun médecin disponible pour la NCT (vacances ou exclu)")

    # Alternance NCT : ne pas répéter le même que la semaine précédente
    if req.last_nct_doctor and req.last_nct_doctor in nct_pool:
        var_nct = x.get((req.last_nct_doctor, 3, "nuit", "NCT"))
        if var_nct is not None:
            model.Add(var_nct == 0)

    # NCT interdit si astreinte nuit la veille (mercredi)
    for doc in nct_pool:
        var_nct = x.get((doc, 3, "nuit", "NCT"))
        var_astreinte_mercredi = x.get((doc, 2, "nuit", "ASTREINTE"))
        if var_nct is not None and var_astreinte_mercredi is not None:
            model.AddImplication(var_nct, var_astreinte_mercredi.Not())

    # --- 6. Fixes forcés (FV) ---
    if fv_id:
        for d_idx, slot, act, forced_val in [
            (0, "nuit", "GARDE", 1),
            (3, "am", "CORO", 1),
        ]:
            var = x.get((fv_id, d_idx, slot, act))
            if var is not None:
                model.Add(var == forced_val)
            else:
                warnings.append(f"FV : créneau {DAY_NAMES_FR[d_idx]} {slot} {act} non disponible")

    # --- 7. Règles d'exclusion métier ---
    # 7.1 AM OFF après garde nuit (lendemain matin)
    for doc_id in medecins_map:
        for d_idx in range(6):
            var_nuit_garde = x.get((doc_id, d_idx, "nuit", "GARDE"))
            if var_nuit_garde is None:
                continue
            am_next_vars = [v for (d, sl, act), v in x.items() if d == d_idx + 1 and sl == "matin" and doc_id == d[0]]
            if am_next_vars:
                presence_matin = model.NewBoolVar(f"presence_matin_{doc_id}_{d_idx+1}")
                model.Add(sum(am_next_vars) >= 1).OnlyEnforceIf(presence_matin)
                model.Add(sum(am_next_vars) == 0).OnlyEnforceIf(presence_matin.Not())
                model.AddImplication(var_nuit_garde, presence_matin.Not())

    # 7.2 Garde nuit => pas d'activité sur AM le même jour
    for doc_id in medecins_map:
        for d_idx in range(7):
            var_nuit_garde = x.get((doc_id, d_idx, "nuit", "GARDE"))
            if var_nuit_garde is None:
                continue
            am_vars = [v for (d, sl, act), v in x.items() if d == d_idx and sl == "am" and doc_id == d[0]]
            if am_vars:
                presence_am = model.NewBoolVar(f"presence_am_{doc_id}_{d_idx}")
                model.Add(sum(am_vars) >= 1).OnlyEnforceIf(presence_am)
                model.Add(sum(am_vars) == 0).OnlyEnforceIf(presence_am.Not())
                model.AddImplication(var_nuit_garde, presence_am.Not())

    # 7.3 Pas d'astreinte nuit si garde ce jour
    for doc_id in medecins_map:
        for d_idx in range(7):
            garde_vars = [v for (d, sl, act), v in x.items() if d == d_idx and doc_id == d[0] and act == "GARDE"]
            if not garde_vars:
                continue
            garde_present = model.NewBoolVar(f"garde_present_{doc_id}_{d_idx}")
            model.Add(sum(garde_vars) >= 1).OnlyEnforceIf(garde_present)
            model.Add(sum(garde_vars) == 0).OnlyEnforceIf(garde_present.Not())
            nuit_astreinte = x.get((doc_id, d_idx, "nuit", "ASTREINTE"))
            if nuit_astreinte is not None:
                model.AddImplication(garde_present, nuit_astreinte.Not())

    # --- 8. Séquences valides pour M, O, W ---
    for doc_id in astreinte_coro_ids:
        for d_idx in range(7):
            m_type = model.NewIntVar(0, 2, f"seq_m_{doc_id}_{d_idx}")
            a_type = model.NewIntVar(0, 2, f"seq_a_{doc_id}_{d_idx}")
            n_type = model.NewIntVar(0, 2, f"seq_n_{doc_id}_{d_idx}")

            var_astr_m = x.get((doc_id, d_idx, "matin", "ASTREINTE"))
            var_garde_m = x.get((doc_id, d_idx, "matin", "GARDE"))
            if var_astr_m is not None:
                model.Add(m_type == 1).OnlyEnforceIf(var_astr_m)
            if var_garde_m is not None:
                model.Add(m_type == 2).OnlyEnforceIf(var_garde_m)
            any_m = model.NewBoolVar(f"any_m_{doc_id}_{d_idx}")
            matin_vars = [v for v in [var_astr_m, var_garde_m] if v is not None]
            if matin_vars:
                model.Add(sum(matin_vars) >= 1).OnlyEnforceIf(any_m)
                model.Add(sum(matin_vars) == 0).OnlyEnforceIf(any_m.Not())
                model.Add(m_type == 0).OnlyEnforceIf(any_m.Not())
            else:
                model.Add(m_type == 0)

            var_astr_a = x.get((doc_id, d_idx, "am", "ASTREINTE"))
            var_garde_a = x.get((doc_id, d_idx, "am", "GARDE"))
            if var_astr_a is not None:
                model.Add(a_type == 1).OnlyEnforceIf(var_astr_a)
            if var_garde_a is not None:
                model.Add(a_type == 2).OnlyEnforceIf(var_garde_a)
            any_a = model.NewBoolVar(f"any_a_{doc_id}_{d_idx}")
            am_vars = [v for v in [var_astr_a, var_garde_a] if v is not None]
            if am_vars:
                model.Add(sum(am_vars) >= 1).OnlyEnforceIf(any_a)
                model.Add(sum(am_vars) == 0).OnlyEnforceIf(any_a.Not())
                model.Add(a_type == 0).OnlyEnforceIf(any_a.Not())
            else:
                model.Add(a_type == 0)

            var_astr_n = x.get((doc_id, d_idx, "nuit", "ASTREINTE"))
            var_garde_n = x.get((doc_id, d_idx, "nuit", "GARDE"))
            if var_astr_n is not None:
                model.Add(n_type == 1).OnlyEnforceIf(var_astr_n)
            if var_garde_n is not None:
                model.Add(n_type == 2).OnlyEnforceIf(var_garde_n)
            any_n = model.NewBoolVar(f"any_n_{doc_id}_{d_idx}")
            nuit_vars = [v for v in [var_astr_n, var_garde_n] if v is not None]
            if nuit_vars:
                model.Add(sum(nuit_vars) >= 1).OnlyEnforceIf(any_n)
                model.Add(sum(nuit_vars) == 0).OnlyEnforceIf(any_n.Not())
                model.Add(n_type == 0).OnlyEnforceIf(any_n.Not())
            else:
                model.Add(n_type == 0)

            model.AddAllowedAssignments([m_type, a_type, n_type], ALLOWED_SEQUENCES)

    # --- 9. Weekend : alternance CH / WOM pour les astreintes ---
    weekend_group = "CH" if req.week_type == 1 else "WOM"

    if weekend_group == "CH":
        # CH sur toutes les astreintes du weekend (Matin, AM, Nuit)
        for d_idx in [5, 6]:  # SAMEDI, DIMANCHE
            for slot in ["matin", "am", "nuit"]:
                var_ch = x.get(("CH", d_idx, slot, "ASTREINTE"))
                if var_ch is not None:
                    model.Add(var_ch == 1)
                # Interdire les autres médecins
                for doc in medecins_map:
                    if doc != "CH":
                        var_other = x.get((doc, d_idx, slot, "ASTREINTE"))
                        if var_other is not None:
                            model.Add(var_other == 0)
    else:
        # WOM sur les astreintes du weekend (Matin, AM, Nuit)
        for d_idx in [5, 6]:
            for slot in ["matin", "am", "nuit"]:
                wom_vars = []
                for doc in wom_pool:
                    var = x.get((doc, d_idx, slot, "ASTREINTE"))
                    if var is not None:
                        wom_vars.append(var)
                # Interdire les autres médecins
                for doc in medecins_map:
                    if doc not in wom_pool:
                        var_other = x.get((doc, d_idx, slot, "ASTREINTE"))
                        if var_other is not None:
                            model.Add(var_other == 0)
                if wom_vars:
                    model.Add(sum(wom_vars) == 1)
                else:
                    warnings.append(f"Weekend {DAY_NAMES_FR[d_idx]} {slot} : aucun médecin W/O/M disponible, CH utilisé")
                    var_ch = x.get(("CH", d_idx, slot, "ASTREINTE"))
                    if var_ch is not None:
                        model.Add(var_ch == 1)

    # --- 10. Préservation des saisies manuelles ---
    if req.existing_schedule:
        for (row_key, day_name), doctors in req.existing_schedule.items():
            slot, activity = map_row_key_to_slot_activity(row_key)
            if slot is None or activity is None:
                continue
            day_idx = DAYS.index(day_name)
            # Forcer les médecins présents à 1
            for doc in doctors:
                var = x.get((doc, day_idx, slot, activity))
                if var is not None:
                    model.Add(var == 1)
            # Forcer les autres à 0
            for doc in medecins_map:
                if doc not in doctors:
                    var = x.get((doc, day_idx, slot, activity))
                    if var is not None:
                        model.Add(var == 0)

    # --- 11. Équité (objectif) ---
    points = {doc: 0 for doc in astreinte_coro_ids}
    for (doc, d_idx, slot, activity), var in x.items():
        if doc in astreinte_coro_ids:
            points[doc] += var

    # Points de weekend (si WOM en weekend, on ajoute les points)
    if req.weekend_mode == "ROTATION":
        # On n'ajoute pas de points supplémentaires ici car ils sont déjà comptés
        pass

    if "M" in points and "W" in points:
        model.Add(points["M"] == points["W"])

    if "O" in points and "M" in points:
        dev_O = model.NewIntVar(0, 10, "dev_O")
        model.Add(dev_O >= points["O"] - points["M"])
        model.Add(dev_O >= points["M"] - points["O"])
        model.Minimize(dev_O)
    else:
        model.Minimize(sum(points.values()))

    # --- 12. Résolution ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)

    # --- 13. Extraction des résultats ---
    assignments: List[Assignment] = []

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for (doc, d_idx, slot, activity), var in x.items():
            if solver.Value(var) == 1:
                assignments.append(Assignment(
                    date=days[d_idx].isoformat(),
                    day_name=DAY_NAMES_FR[d_idx],
                    slot=slot,
                    activity=activity,
                    doctor=doc,
                    note="assigné par le solveur"
                ))

        # Ajouter les CH pour les nuits structurelles si manquants
        # On vérifie si CH est présent pour les jours où il est attendu
        for d_idx in range(5):
            if structure[d_idx] == "CH":
                already = any(a.date == days[d_idx].isoformat() and a.slot == "nuit" and a.doctor == "CH" for a in assignments)
                if not already:
                    assignments.append(Assignment(
                        date=days[d_idx].isoformat(),
                        day_name=DAY_NAMES_FR[d_idx],
                        slot="nuit",
                        activity="ASTREINTE",
                        doctor="CH",
                        note="Structure fixe CH"
                    ))
            else:
                # Vérifier qu'il y a au moins un médecin WOM
                already = any(a.date == days[d_idx].isoformat() and a.slot == "nuit" and a.doctor in wom_pool for a in assignments)
                if not already:
                    assignments.append(Assignment(
                        date=days[d_idx].isoformat(),
                        day_name=DAY_NAMES_FR[d_idx],
                        slot="nuit",
                        activity="ASTREINTE",
                        doctor="CH",
                        note="Fallback CH (aucun WOM disponible)"
                    ))

        # Ajouter les weekends
        for d_idx in [5, 6]:
            day_name = DAY_NAMES_FR[d_idx]
            for slot in ["matin", "am", "nuit"]:
                # Vérifier si une assignation existe déjà pour ce créneau
                already = any(a.date == days[d_idx].isoformat() and a.slot == slot for a in assignments)
                if not already:
                    if weekend_group == "CH":
                        doctor = "CH"
                        note = "Weekend CH (structure)"
                    else:
                        # Trouver un médecin WOM disponible pour le fallback
                        available = [doc for doc in wom_pool if not is_on_vacation(doc, days[d_idx], req.vacations)]
                        if available:
                            doctor = available[0]
                            note = "Weekend WOM (fallback)"
                        else:
                            doctor = "CH"
                            note = "Weekend CH (fallback)"
                    assignments.append(Assignment(
                        date=days[d_idx].isoformat(),
                        day_name=day_name,
                        slot=slot,
                        activity="ASTREINTE",
                        doctor=doctor,
                        note=note
                    ))

    else:
        warnings.append("Aucune solution trouvée par le solveur")

    # Trier
    assignments.sort(key=lambda a: (a.date, SLOTS.index(a.slot) if a.slot in SLOTS else 999))

    return GenerateWeekResponse(
        week_start_date=req.week_start_date,
        assignments=assignments,
        warnings=warnings
    )