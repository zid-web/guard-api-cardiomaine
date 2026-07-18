"""
Solveur de generation des gardes/astreintes - Planning Cardiomaine
Implemente toutes les regles definies :
- Equite generale (tous les medecins)
- Equite astreinte (uniquement W, M, O)
- Structure fixe semaine type 1 / type 2 (alternance)
- CH = mutualisation externe (Centre Hospitalier)
- FV = medecin externe, garde fixe lundi nuit + coro jeudi apres-midi
- NCT jeudi : alternance stricte W/M, jamais la veille d'astreinte nuit
- Astreinte weekend : alternance CH vs rotation M/O/W (weekend entier)
- Exclusion automatique des medecins en vacances
"""

from ortools.sat.python import cp_model
from datetime import date, timedelta
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel


# ============================================
# MODELES DE DONNEES (entree/sortie de l'API)
# ============================================

class Vacation(BaseModel):
    doctor_id: str
    start_date: str  # format YYYY-MM-DD
    end_date: str


class DoctorEquity(BaseModel):
    doctor_id: str
    astreinte_count: int = 0   # nombre d'astreintes deja faites (cumul historique)
    nct_count: int = 0
    weekend_count: int = 0


class GenerateWeekRequest(BaseModel):
    week_start_date: str          # Lundi de la semaine, format YYYY-MM-DD
    week_type: Literal[1, 2]      # 1 = semaine type 1 (Mer/Jeu fixes), 2 = inversee
    weekend_mode: Literal["CH", "ROTATION"]  # alternance weekend, fournie par l'appelant
    vacations: List[Vacation] = []
    equity: List[DoctorEquity] = []   # compteurs actuels, pour respecter l'equite
    last_nct_doctor: Optional[str] = None  # qui a fait la NCT la semaine precedente (W ou M)

    # Listes de medecins (modifiable sans redeployer le solveur)
    astreinte_pool: List[str] = ["W", "M", "O"]
    all_doctors: List[str] = ["A", "Z", "S", "B", "G", "O", "W", "M", "P", "H", "U", "K", "V"]
    nct_pool: List[str] = ["W", "M"]


class Assignment(BaseModel):
    date: str
    day_name: str
    slot: str          # "nuit", "matin", "apres_midi", "weekend"
    activity: str       # "Astreinte Nuit", "NCT", "Garde Nuit", "Coro Apres-midi", "Astreinte Weekend"
    doctor: str          # code medecin ou "CH"
    note: Optional[str] = None


class GenerateWeekResponse(BaseModel):
    week_start_date: str
    week_type: int
    assignments: List[Assignment]
    warnings: List[str] = []


# ============================================
# UTILITAIRES
# ============================================

DAY_NAMES_FR = ["LUNDI", "MARDI", "MERCREDI", "JEUDI", "VENDREDI", "SAMEDI", "DIMANCHE"]


def is_on_vacation(doctor_id: str, day: date, vacations: List[Vacation]) -> bool:
    for v in vacations:
        if v.doctor_id != doctor_id:
            continue
        start = date.fromisoformat(v.start_date)
        end = date.fromisoformat(v.end_date)
        if start <= day <= end:
            return True
    return False


def get_equity_count(doctor_id: str, equity: List[DoctorEquity], field: str) -> int:
    for e in equity:
        if e.doctor_id == doctor_id:
            return getattr(e, field)
    return 0


# ============================================
# SOLVEUR PRINCIPAL
# ============================================

def generate_week(req: GenerateWeekRequest) -> GenerateWeekResponse:
    warnings: List[str] = []
    week_start = date.fromisoformat(req.week_start_date)
    days = [week_start + timedelta(days=i) for i in range(7)]  # Lun..Dim

    model = cp_model.CpModel()
    assignments: List[Assignment] = []

    # ---- Variables : astreinte de nuit, Lundi a Vendredi ----
    # Pour chaque jour de semaine (0=Lundi..4=Vendredi), une variable qui vaut
    # soit "CH" soit un des medecins du pool astreinte (W,M,O), selon la structure fixe.
    weekday_night_doctor: Dict[int, cp_model.IntVar] = {}
    pool = req.astreinte_pool  # ["W","M","O"]
    pool_index = {d: i for i, d in enumerate(pool)}
    # index -1 reserve pour CH
    CH_CODE = -1

    def eligible_pool_for(day_idx: int) -> List[str]:
        """Retourne la liste des medecins eligibles (non en vacances) pour ce jour."""
        d = days[day_idx]
        return [doc for doc in pool if not is_on_vacation(doc, d, req.vacations)]

    # Determine la structure fixe (CH ou pool) selon le type de semaine
    # index jours : 0=Lundi 1=Mardi 2=Mercredi 3=Jeudi 4=Vendredi
    if req.week_type == 1:
        fixed_structure = {0: "CH", 1: "CH", 2: "POOL", 3: "POOL", 4: "CH"}
    else:
        fixed_structure = {0: "POOL", 1: "POOL", 2: "CH", 3: "CH", 4: "POOL"}

    night_vars: Dict[int, cp_model.IntVar] = {}
    for day_idx in range(5):  # Lundi a Vendredi
        if fixed_structure[day_idx] == "CH":
            # Jour fixe CH, pas de variable necessaire
            assignments.append(Assignment(
                date=days[day_idx].isoformat(),
                day_name=DAY_NAMES_FR[day_idx],
                slot="nuit",
                activity="Astreinte Nuit",
                doctor="CH",
                note="Mutualisation Centre Hospitalier (structure fixe)"
            ))
        else:
            elig = eligible_pool_for(day_idx)
            if not elig:
                warnings.append(f"{DAY_NAMES_FR[day_idx]} {days[day_idx]}: aucun medecin W/M/O disponible (tous en vacances) - CH par defaut")
                assignments.append(Assignment(
                    date=days[day_idx].isoformat(), day_name=DAY_NAMES_FR[day_idx],
                    slot="nuit", activity="Astreinte Nuit", doctor="CH",
                    note="Aucun medecin disponible, bascule CH automatique"
                ))
                continue
            var = model.NewIntVarFromDomain(
                cp_model.Domain.FromValues([pool_index[d] for d in elig]),
                f"night_{day_idx}"
            )
            night_vars[day_idx] = var

    # ---- Variable NCT (jeudi uniquement, W ou M) ----
    nct_var = None
    thursday = days[3]
    nct_elig = [d for d in req.nct_pool if not is_on_vacation(d, thursday, req.vacations)]
    # Ne jamais reprendre la meme personne que la semaine precedente si evitable
    if req.last_nct_doctor in nct_elig and len(nct_elig) > 1:
        nct_elig_preferred = [d for d in nct_elig if d != req.last_nct_doctor]
    else:
        nct_elig_preferred = nct_elig

    if nct_elig_preferred:
        nct_pool_index = {d: i for i, d in enumerate(req.nct_pool)}
        nct_var = model.NewIntVarFromDomain(
            cp_model.Domain.FromValues([nct_pool_index[d] for d in nct_elig_preferred]),
            "nct"
        )
    else:
        warnings.append("JEUDI: aucun medecin W/M disponible pour la NCT (vacances) - a assigner manuellement")

    # ---- Contrainte : le medecin de NCT ne peut pas etre d'astreinte nuit le MERCREDI (veille) ----
    if nct_var is not None and 2 in night_vars:  # 2 = index mercredi
        wed_var = night_vars[2]
        for name in nct_elig_preferred:
            nct_val = req.nct_pool.index(name)
            if name in pool:  # le medecin NCT existe aussi dans le pool astreinte
                wed_val = pool_index[name]
                # Si nct_var == nct_val ALORS wed_var != wed_val
                b = model.NewBoolVar(f"nct_is_{name}")
                model.Add(nct_var == nct_val).OnlyEnforceIf(b)
                model.Add(nct_var != nct_val).OnlyEnforceIf(b.Not())
                model.Add(wed_var != wed_val).OnlyEnforceIf(b)

    # ---- Objectif d'equite : minimiser la charge cumulee (favorise les moins solicites) ----
    objective_terms = []
    for day_idx, var in night_vars.items():
        for doc in pool:
            if pool_index[doc] > len(pool) - 1:
                continue
            b = model.NewBoolVar(f"eq_night_{day_idx}_{doc}")
            model.Add(var == pool_index[doc]).OnlyEnforceIf(b)
            model.Add(var != pool_index[doc]).OnlyEnforceIf(b.Not())
            cost = get_equity_count(doc, req.equity, "astreinte_count")
            objective_terms.append(cost * b)

    if objective_terms:
        model.Minimize(sum(objective_terms))

    # ---- Resolution ----
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for day_idx, var in night_vars.items():
            chosen = pool[solver.Value(var)]
            assignments.append(Assignment(
                date=days[day_idx].isoformat(), day_name=DAY_NAMES_FR[day_idx],
                slot="nuit", activity="Astreinte Nuit", doctor=chosen,
                note="Assignation par equite (moins solicite en priorite)"
            ))
        if nct_var is not None:
            chosen_nct = req.nct_pool[solver.Value(nct_var)]
            assignments.append(Assignment(
                date=thursday.isoformat(), day_name="JEUDI",
                slot="nuit", activity="NCT", doctor=chosen_nct,
                note="Alternance W/M, distincte de l'astreinte nuit"
            ))
    else:
        warnings.append("Aucune solution trouvee automatiquement pour l'astreinte de semaine - assignation manuelle requise")

    # ---- FV : garde nuit fixe lundi, coro fixe jeudi apres-midi (sauf vacances) ----
    monday = days[0]
    if is_on_vacation("FV", monday, req.vacations):
        warnings.append("LUNDI: FV est en conges, garde de nuit du lundi a reassigner manuellement")
        assignments.append(Assignment(
            date=monday.isoformat(), day_name="LUNDI", slot="nuit",
            activity="Garde Nuit", doctor="NON_ASSIGNE",
            note="FV en conges - reassignation manuelle necessaire"
        ))
    else:
        assignments.append(Assignment(
            date=monday.isoformat(), day_name="LUNDI", slot="nuit",
            activity="Garde Nuit", doctor="FV", note="FV fixe (garde de nuit tous les lundis)"
        ))

    if is_on_vacation("FV", thursday, req.vacations):
        warnings.append("JEUDI: FV est en conges, coro apres-midi a reassigner manuellement")
        assignments.append(Assignment(
            date=thursday.isoformat(), day_name="JEUDI", slot="apres_midi",
            activity="Coro Apres-midi", doctor="NON_ASSIGNE",
            note="FV en conges - reassignation manuelle necessaire"
        ))
    else:
        assignments.append(Assignment(
            date=thursday.isoformat(), day_name="JEUDI", slot="apres_midi",
            activity="Coro Apres-midi", doctor="FV", note="FV fixe (coro tous les jeudis apres-midi)"
        ))

    # ---- Weekend : CH entier OU rotation M/O/W (weekend entier), selon weekend_mode ----
    saturday, sunday = days[5], days[6]
    if req.weekend_mode == "CH":
        for d, name in [(saturday, "SAMEDI"), (sunday, "DIMANCHE")]:
            assignments.append(Assignment(
                date=d.isoformat(), day_name=name, slot="weekend",
                activity="Astreinte Weekend", doctor="CH",
                note="Mutualisation Centre Hospitalier (weekend entier)"
            ))
    else:
        elig_weekend = [
            doc for doc in pool
            if not is_on_vacation(doc, saturday, req.vacations)
            and not is_on_vacation(doc, sunday, req.vacations)
        ]
        if not elig_weekend:
            warnings.append("WEEKEND: aucun medecin W/M/O disponible - CH par defaut")
            for d, name in [(saturday, "SAMEDI"), (sunday, "DIMANCHE")]:
                assignments.append(Assignment(
                    date=d.isoformat(), day_name=name, slot="weekend",
                    activity="Astreinte Weekend", doctor="CH",
                    note="Aucun medecin disponible, bascule CH automatique"
                ))
        else:
            # Choix par equite (le moins solicite en weekend)
            chosen = min(elig_weekend, key=lambda d: get_equity_count(d, req.equity, "weekend_count"))
            for d, name in [(saturday, "SAMEDI"), (sunday, "DIMANCHE")]:
                assignments.append(Assignment(
                    date=d.isoformat(), day_name=name, slot="weekend",
                    activity="Astreinte Weekend", doctor=chosen,
                    note="Proposition par equite - MODIFIABLE par l'admin"
                ))

    # Trier par date
    assignments.sort(key=lambda a: a.date)

    return GenerateWeekResponse(
        week_start_date=req.week_start_date,
        week_type=req.week_type,
        assignments=assignments,
        warnings=warnings,
    )
