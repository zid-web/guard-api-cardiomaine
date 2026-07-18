"""
Solveur complet pour le planning Cardiomaine - Version finale
Toutes les contraintes métier sont implémentées.
"""

from ortools.sat.python import cp_model
from datetime import date, timedelta
from typing import List, Dict, Optional, Literal, Set, Tuple
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
    # Compteurs d'équité historiques
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
    medecins: List[Medecin]
    vacations: List[Vacation] = []
    weekend_mode: Literal["CH", "ROTATION"]
    semaine_iso_impaire: bool = False   # True = SB, False = SA
    last_nct_doctor: Optional[str] = None  # W ou M

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
# On interdit tout cumul garde (2) et astreinte (1) sur la même journée
# 0 = rien (planning / consultation), 1 = ASTREINTE, 2 = GARDE
ALLOWED_SEQUENCES = [
    (0, 0, 0),  # rien
    (1, 1, 1),  # que des astreintes
    (2, 2, 2),  # que des gardes
    (0, 1, 1),  # planning + astreinte AM et nuit
    (0, 2, 2),  # planning + garde AM et nuit
    (1, 1, 0),  # astreinte matin et AM, pas de nuit
    (2, 2, 0),  # garde matin et AM, pas de nuit
    (1, 0, 1),  # astreinte matin et nuit, pas d'AM
    (2, 0, 2),  # garde matin et nuit, pas d'AM
    (0, 1, 0),  # astreinte AM uniquement
    (0, 2, 0),  # garde AM uniquement
    (1, 0, 0),  # astreinte matin uniquement
    (2, 0, 0),  # garde matin uniquement
    (0, 0, 1),  # astreinte nuit uniquement
    (0, 0, 2),  # garde nuit uniquement
]
# Note : on exclut les combinaisons mixtes (1 et 2 ensemble) comme (1,2,0) etc.

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

# ============================================================
# 3. SOLVEUR PRINCIPAL
# ============================================================

def generate_week(req: GenerateWeekRequest) -> GenerateWeekResponse:
    warnings = []
    week_start = date.fromisoformat(req.week_start_date)
    days = jours_semaine(week_start)

    # --- 1. Préparation des données ---
    medecins_map = {m.id: m for m in req.medecins}
    astreinte_coro_ids = {m.id for m in req.medecins if m.statut == StatutMedecin.ASTREINTE_CORO}
    nct_pool = {m.id for m in req.medecins if m.statut == StatutMedecin.ASTREINTE_CORO and m.id != "O"}
    permanent_ids = {m.id for m in req.medecins if m.statut == StatutMedecin.PERMANENT}
    fv_id = next((m.id for m in req.medecins if m.statut == StatutMedecin.FV), None)
    daas_id = next((m.id for m in req.medecins if m.statut == StatutMedecin.DAAS), None)
    d_id = next((m.id for m in req.medecins if m.statut == StatutMedecin.D), None)

    # Demi-journées libres (jour, slot) -> set de médecins
    half_days_off = {
        ("MERCREDI", "am"): {"M", "W", "G", "Z", "H"},
        ("JEUDI", "am"): {"U", "S", "P"},
        ("VENDREDI", "am"): {"O", "A", "K", "R", "T"},
    }

    # Fixes exclusifs (jours où le médecin ne fait PAS de garde/astreinte)
    # Maintenant, on exclut seulement pour les jours où c'est explicitement "pas de garde"
    # Mais on garde les exclusions pour P, U, A, S sur leurs jours de consultation fixe,
    # car ils ne doivent pas faire de garde/astreinte ces jours-là (règle initiale).
    fixed_exclusions = {
        "P": {1},        # mardi
        "U": {2},        # mercredi
        "A": {0, 3},     # lundi, jeudi
        "S": {0, 4},     # lundi, vendredi
    }

    # --- 2. Création des variables ---
    model = cp_model.CpModel()
    x = {}  # clé: (doc, day_idx, slot, activity) -> BoolVar

    def add_var_if_allowed(doc_id: str, d_idx: int, slot: str, activity: str):
        day = days[d_idx]
        if is_on_vacation(doc_id, day, req.vacations):
            return

        # Exclure Daas et D de tout
        if doc_id in (daas_id, d_id):
            return

        # Exclure FV sauf ses créneaux spécifiques
        if doc_id == fv_id:
            if not (d_idx == 0 and slot == "nuit" and activity == "GARDE") and \
               not (d_idx == 3 and slot == "am" and activity == "CORO"):
                return

        # Exclure les demi-journées libres
        day_name = DAY_NAMES_FR[d_idx]
        if day_name in half_days_off and slot in half_days_off[day_name]:
            if doc_id in half_days_off[day_name][slot]:
                return

        # Exclure les fixés (P, U, A, S)
        if doc_id in fixed_exclusions and d_idx in fixed_exclusions[doc_id]:
            return

        # Restrictions spécifiques par statut
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
        if statut == StatutMedecin.FV:
            # déjà restreint plus haut
            pass

        var = model.NewBoolVar(f"x_{doc_id}_{d_idx}_{slot}_{activity}")
        x[(doc_id, d_idx, slot, activity)] = var

    for doc_id in medecins_map:
        for d_idx in range(7):
            for slot in SLOTS:
                for activity in ACTIVITIES:
                    add_var_if_allowed(doc_id, d_idx, slot, activity)

    # --- 3. Contraintes générales ---

    # 3.1 Capacité : max 2 médecins par case (jour, slot)
    for d_idx in range(7):
        for slot in SLOTS:
            case_vars = [v for (d, sl, act), v in x.items() if d == d_idx and sl == slot]
            if case_vars:
                model.Add(sum(case_vars) <= 2)

    # 3.2 Unicité : un médecin ne peut faire qu'une activité par créneau
    for doc_id in medecins_map:
        for d_idx in range(7):
            for slot in SLOTS:
                slot_vars = [v for (d, sl, act), v in x.items() if d == d_idx and sl == slot and doc_id == d[0]]
                if slot_vars:
                    model.Add(sum(slot_vars) <= 1)

    # --- 4. Structure SA/SB pour les nuits de semaine (0..4) ---
    if req.semaine_iso_impaire:  # SB
        structure = {0: "CH", 1: "CH", 2: "POOL", 3: "POOL", 4: "CH"}
    else:  # SA
        structure = {0: "POOL", 1: "POOL", 2: "CH", 3: "CH", 4: "POOL"}

    for d_idx in range(5):
        nuit_astreinte_vars = [v for (d, sl, act), v in x.items() if d == d_idx and sl == "nuit" and act == "ASTREINTE"]
        if structure[d_idx] == "CH":
            for var in nuit_astreinte_vars:
                model.Add(var == 0)
        else:
            # Seuls M, O, W peuvent faire l'astreinte nuit ce jour
            for (doc, d, sl, act), var in x.items():
                if d == d_idx and sl == "nuit" and act == "ASTREINTE":
                    if doc not in astreinte_coro_ids:
                        model.Add(var == 0)
            # Au moins un médecin du pool
            pool_vars = [v for (doc, d, sl, act), v in x.items() if d == d_idx and sl == "nuit" and act == "ASTREINTE" and doc in astreinte_coro_ids]
            if pool_vars:
                model.Add(sum(pool_vars) >= 1)
            else:
                warnings.append(f"Jour {DAY_NAMES_FR[d_idx]} : aucun médecin du pool disponible pour l'astreinte nuit, CH sera utilisé.")

    # --- 5. NCT (jeudi nuit) ---
    nct_vars = [v for (doc, d, sl, act), v in x.items() if d == 3 and sl == "nuit" and act == "NCT"]
    if nct_vars:
        model.Add(sum(nct_vars) == 1)
    else:
        warnings.append("JEUDI : aucun médecin disponible pour la NCT (vacances ou exclu)")

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
                warnings.append(f"FV : créneau {DAY_NAMES_FR[d_idx]} {slot} {act} non disponible (vacances ou conflit)")

    # --- 7. Règles d'exclusion métier ---

    # 7.1 AM OFF après garde nuit (lendemain matin)
    for doc_id in medecins_map:
        for d_idx in range(6):  # jusqu'à vendredi
            var_nuit_garde = x.get((doc_id, d_idx, "nuit", "GARDE"))
            if var_nuit_garde is None:
                continue
            am_next_vars = [v for (d, sl, act), v in x.items() if d == d_idx + 1 and sl == "matin" and doc_id == d[0]]
            if am_next_vars:
                presence_matin = model.NewBoolVar(f"presence_matin_{doc_id}_{d_idx+1}")
                model.Add(sum(am_next_vars) >= 1).OnlyEnforceIf(presence_matin)
                model.Add(sum(am_next_vars) == 0).OnlyEnforceIf(presence_matin.Not())
                model.AddImplication(var_nuit_garde, presence_matin.Not())

    # 7.2 Garde nuit => pas d'activité sur AM le même jour (après-midi libre)
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

    # 7.3 Pas d'astreinte nuit si garde ce jour (même médecin)
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

    # --- 8. Séquences valides pour M, O, W (pas de cumul garde/astreinte) ---
    # On crée des variables entières pour matin, am, nuit
    for doc_id in astreinte_coro_ids:
        for d_idx in range(7):
            m_type = model.NewIntVar(0, 2, f"seq_m_{doc_id}_{d_idx}")
            a_type = model.NewIntVar(0, 2, f"seq_a_{doc_id}_{d_idx}")
            n_type = model.NewIntVar(0, 2, f"seq_n_{doc_id}_{d_idx}")

            # Liaison avec les booléens
            def link_type(var_bool, target_var, val):
                if var_bool is not None:
                    model.Add(target_var == val).OnlyEnforceIf(var_bool)

            # Matin
            var_astr_m = x.get((doc_id, d_idx, "matin", "ASTREINTE"))
            var_garde_m = x.get((doc_id, d_idx, "matin", "GARDE"))
            if var_astr_m is not None:
                model.Add(m_type == 1).OnlyEnforceIf(var_astr_m)
            if var_garde_m is not None:
                model.Add(m_type == 2).OnlyEnforceIf(var_garde_m)
            # Si aucun, m_type = 0
            any_m = model.NewBoolVar(f"any_m_{doc_id}_{d_idx}")
            matin_vars = [v for v in [var_astr_m, var_garde_m] if v is not None]
            if matin_vars:
                model.Add(sum(matin_vars) >= 1).OnlyEnforceIf(any_m)
                model.Add(sum(matin_vars) == 0).OnlyEnforceIf(any_m.Not())
                model.Add(m_type == 0).OnlyEnforceIf(any_m.Not())
            else:
                model.Add(m_type == 0)

            # AM
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

            # Nuit
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

            # Contrainte de table
            model.AddAllowedAssignments([m_type, a_type, n_type], ALLOWED_SEQUENCES)

    # --- 9. Weekend ---
    weekend_docs = []
    if req.weekend_mode == "CH":
        pass
    else:
        weekend_docs = [doc for doc in astreinte_coro_ids
                        if not is_on_vacation(doc, days[5], req.vacations)
                        and not is_on_vacation(doc, days[6], req.vacations)]
        if not weekend_docs:
            warnings.append("WEEKEND : aucun médecin du pool disponible, CH utilisé")
            weekend_docs = ["CH"]

    # --- 10. Équité (objectif) ---
    points = {doc: 0 for doc in astreinte_coro_ids}
    for (doc, d_idx, slot, activity), var in x.items():
        if doc in astreinte_coro_ids:
            points[doc] += var

    # Points de weekend
    if req.weekend_mode == "ROTATION" and weekend_docs and weekend_docs[0] != "CH":
        for doc in weekend_docs:
            if doc in points:
                points[doc] += 2  # samedi + dimanche

    # Contrainte d'équité stricte : M = W
    if "M" in points and "W" in points:
        model.Add(points["M"] == points["W"])

    # Objectif : minimiser l'écart de O par rapport à M (ou W)
    if "O" in points and "M" in points:
        dev_O = model.NewIntVar(0, 10, "dev_O")
        model.Add(dev_O >= points["O"] - points["M"])
        model.Add(dev_O >= points["M"] - points["O"])
        model.Minimize(dev_O)
    else:
        model.Minimize(sum(points.values()))

    # --- 11. Résolution ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)

    # --- 12. Extraction des résultats ---
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

        # Ajouter les CH pour les nuits structurelles
        for d_idx in range(5):
            if structure[d_idx] == "CH":
                already = any(a.date == days[d_idx].isoformat() and a.slot == "nuit" for a in assignments)
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
                already = any(a.date == days[d_idx].isoformat() and a.slot == "nuit" and a.activity == "ASTREINTE" for a in assignments)
                if not already:
                    assignments.append(Assignment(
                        date=days[d_idx].isoformat(),
                        day_name=DAY_NAMES_FR[d_idx],
                        slot="nuit",
                        activity="ASTREINTE",
                        doctor="CH",
                        note="Aucun médecin disponible, CH par défaut"
                    ))

        # Ajouter les weekends
        if req.weekend_mode == "CH":
            for d_idx in [5, 6]:
                assignments.append(Assignment(
                    date=days[d_idx].isoformat(),
                    day_name=DAY_NAMES_FR[d_idx],
                    slot="weekend",
                    activity="ASTREINTE",
                    doctor="CH",
                    note="Weekend CH"
                ))
        else:
            if weekend_docs and weekend_docs[0] != "CH":
                doc_weekend = weekend_docs[0]
                for d_idx in [5, 6]:
                    assignments.append(Assignment(
                        date=days[d_idx].isoformat(),
                        day_name=DAY_NAMES_FR[d_idx],
                        slot="weekend",
                        activity="ASTREINTE",
                        doctor=doc_weekend,
                        note="Weekend en rotation (équité)"
                    ))
            else:
                for d_idx in [5, 6]:
                    assignments.append(Assignment(
                        date=days[d_idx].isoformat(),
                        day_name=DAY_NAMES_FR[d_idx],
                        slot="weekend",
                        activity="ASTREINTE",
                        doctor="CH",
                        note="Weekend CH (fallback)"
                    ))

    else:
        warnings.append("Aucune solution trouvée par le solveur")

    assignments.sort(key=lambda a: (a.date, SLOTS.index(a.slot) if a.slot in SLOTS else 999))

    return GenerateWeekResponse(
        week_start_date=req.week_start_date,
        assignments=assignments,
        warnings=warnings
    )