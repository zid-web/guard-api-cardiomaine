"""
Solveur complet pour le planning Cardiomaine - Version avec alternance CH / WOM
et préservation des saisies manuelles.
"""

from ortools.sat.python import cp_model
from datetime import date, timedelta
from typing import List, Dict, Optional, Literal, Set, Tuple, Any
from pydantic import BaseModel
import enum
import re

from config import load_default_rules, merge_rules, half_days_off_as_dict, fixed_exclusions_as_dict

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
    points_coro: int = 0  # Équité CORO : scope MENSUEL (pas cumulé à vie comme les autres
                           # catégories), pertinent uniquement pour M/O/W - voir plus bas.
    # Pondération d'équité : 100 = charge cible normale (plein temps, sans ajustement
    # d'ancienneté). Une valeur < 100 réduit la charge cible de ce médecin (ex: 50 pour
    # un mi-temps, 70 pour un médecin senior dont la charge est volontairement allégée).
    # Une valeur > 100 est possible mais rare (ex: augmenter temporairement la charge
    # cible d'un médecin qui "doit" plus aux autres après une longue absence passée).
    # Couvre à la fois le temps partiel ET l'ancienneté : un seul curseur, pas deux
    # mécanismes séparés qui se chevaucheraient.
    poids_equite_pct: int = 100

class Vacation(BaseModel):
    doctor_id: str
    start_date: str
    end_date: str

class GenerateWeekRequest(BaseModel):
    week_start_date: str              # YYYY-MM-DD (lundi)
    week_type: int                    # 1 = impaire, 2 = paire
    medecins: List[Medecin]
    vacations: List[Vacation] = []
    congres: List[Vacation] = []      # même structure que vacations : doctor_id, start_date, end_date
    weekend_mode: Literal["CH", "ROTATION"] = "ROTATION"
    last_nct_doctor: Optional[str] = None  # W ou M
    previous_sunday_guard_doctor: Optional[str] = None  # médecin ayant fait la garde/astreinte
                                                          # de nuit dimanche (semaine précédente) ;
                                                          # sert à forcer son 1/2 journée off ce lundi
                                                          # (règle métier: "Garde de nuit dimanche ->
                                                          # 1/2 journée off lundi, appliquée systématiquement")
    existing_schedule: Optional[Dict[str, List[str]]] = None  # clé "row_key||day_name" -> [doctors]
    rules_override: Optional[Dict[str, Any]] = None  # surcharge partielle de rules_config.json, sans redéploiement
    historical_patterns: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
    # { row_key: { day_name: { eligible_doctors: [str], frequency: {doctor_id: count} } } }
    # Activités hors périmètre du solveur (Cs, ETT, EE, hors site...) - construit
    # côté front (lib/pattern-analysis.ts) à partir de l'historique réel.
    # eligible_doctors est DÉDUIT des données (qui a déjà fait cette activité ce
    # jour-là par le passé), pas une liste figée à la main comme REEDUC_ALLOWED.

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
ACTIVITIES = ["ASTREINTE", "GARDE", "CORO", "NCT", "REEDUC", "PRE_OP", "RYTHMO"]

# Les listes d'éligibilité par médecin (REEDUC_ALLOWED, CORO_ALLOWED, RYTHMO_ALLOWED,
# NCT_ALLOWED), les demi-journées off, les exclusions fixes et le calendrier NCT figé
# ne sont plus codés en dur ici : ils sont chargés depuis rules_config.json (+ surcharge
# optionnelle envoyée par le front dans req.rules_override). Voir config.py.

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
        "Reeduc": ("am", "REEDUC"),          # hypothèse : créneau après-midi, à confirmer
        "Apm - RÉEDUCATION": ("am", "REEDUC"),
        "Pré-op": ("am", "PRE_OP"),
        "Matin - Rythmo": ("matin", "RYTHMO"),   # hypothèse : même structure que CORO (2 lignes), à confirmer
        "Apm - Rythmo": ("am", "RYTHMO"),
    }
    return mapping.get(row_key, (None, None))


def map_historical_row_key_to_slot(row_key: str) -> Optional[str]:
    """Mapper générique pour les activités hors solveur restant basées sur la
    fréquence historique (Cs, ETT, EE, Stress...). Les activités "Hors site - X"
    ont leurs propres règles fixes (voir HORS_SITE_CONFIG) - pas de mapping
    générique ici pour elles, gérées séparément.
    """
    if not is_solver_historical_row_key(row_key):
        return None
    if row_key.startswith("Matin - "):
        return "matin"
    if row_key.startswith("Apm - "):
        return "am"
    return None


# Lignes structurelles / hors fidélité historique (½-off, Visite, hors site,
# solveur classique...) : ignorées silencieusement dans historical_patterns,
# pas de warning "non reconnu" pour elles - elles sont gérées ailleurs dans
# le solveur ou ne concernent pas la fidélité historique.
HISTORICAL_PATTERNS_SKIP_ROW_KEYS = {
    "1/2 journée off Matin",
    "1/2 journée off Après-midi",
    "Matin - Visite",
    "Hors site - CDL",
    "Hors site - IRM",
    "Hors site - Scinti",
    "Hors site - LFB",
    "Hors site - PSSL",
    "Apm - RÉEDUCATION",
    "Pré-op",
    "Entrées PSS",
    "Notes du jour",
    "Congrès",
    "Congés",
    "Vacances",
    "Matin - Rythmo",
    "Apm - Rythmo",
    "Matin - Coro",
    "Apm - Coro",
    "Garde Matin",
    "Garde Midi",
    "Garde Nuit",
    "Astreintes ATL Matin",
    "Astreintes ATL Midi",
    "Astreintes ATL Nuit",
    "Hors site - NCT",
}


def is_solver_historical_row_key(row_key: str) -> bool:
    """Allowlist Cs / ETT / EE / Stress uniquement (fidélité historique) -
    tout le reste (activités déjà gérées par des règles dures, ou hors
    périmètre) est explicitement exclu plutôt que deviné par préfixe seul."""
    if row_key in HISTORICAL_PATTERNS_SKIP_ROW_KEYS or row_key in HORS_SITE_CONFIG:
        return False
    return bool(re.match(r"^(Matin|Apm) - (Cs |ETT |EE\d|Stress$)", row_key))


# Règles fixes hors site (confirmées avec l'utilisateur le 26/07/2026, PAS
# déduites de l'historique comme Cs/ETT/EE) :
# - IRM, CDL, Scinti : ne bloquent que le MATIN (le médecin reste disponible
#   l'après-midi pour d'autres activités).
# - LFB, PSSL : bloquent la JOURNÉE ENTIÈRE (matin ET après-midi).
# - CDL : H toujours prioritaire ; O uniquement si H est déjà pris ailleurs ce
#   jour-là (pas seulement en vacances - géré via priorité dans l'objectif,
#   pas via une exclusion a priori, pour laisser le solveur choisir O si H est
#   occupé par autre chose ce jour précis).
HORS_SITE_CONFIG = {
    "Hors site - IRM":    {"allowed": ["S"],           "full_day": False},
    "Hors site - Scinti": {"allowed": ["R", "T"],      "full_day": False},
    "Hors site - CDL":    {"allowed": ["V", "O"],      "full_day": False, "priority": ["V", "O"]},
    "Hors site - LFB":    {"allowed": ["H", "S", "G"], "full_day": True},
    "Hors site - PSSL":   {"allowed": ["Z", "B"],      "full_day": True},
}

# ============================================================
# 3. SOLVEUR PRINCIPAL
# ============================================================

def generate_week(req: GenerateWeekRequest) -> GenerateWeekResponse:
    warnings = []
    week_start = date.fromisoformat(req.week_start_date)
    days = jours_semaine(week_start)

    # --- 0. Chargement des règles (défaut JSON + surcharge éventuelle du front) ---
    rules = merge_rules(load_default_rules(), req.rules_override)
    REEDUC_ALLOWED = set(rules["reeduc_allowed"])
    REEDUC_DAYS = set(rules["reeduc_days"])
    CORO_ALLOWED = set(rules["coro_allowed"])
    RYTHMO_ALLOWED = set(rules["rythmo_allowed"])
    RYTHMO_SCHEDULE: Dict[str, set] = {doc: set(d) for doc, d in rules.get("rythmo_schedule", {}).items()}
    # Créneaux (matin/am) par médecin, séparés du jour - permet à chacun d'avoir
    # un créneau différent (P=matin+am, U/A=am seul). Défaut matin+am si absent
    # du JSON, pour rester rétrocompatible avec d'anciennes configs.
    RYTHMO_SLOTS: Dict[str, Tuple[str, ...]] = {
        doc: tuple(slots) for doc, slots in rules.get("rythmo_slots", {}).items()
    }
    # Vendredi : alternance P/U selon la semaine (confirmé utilisateur 27/07/2026).
    # Traité à PART du mécanisme générique RYTHMO_SCHEDULE/RYTHMO_SLOTS (qui suppose
    # un créneau uniforme par médecin sur tous ses jours) car P a besoin de
    # matin+am le mardi mais SEULEMENT matin le vendredi - un seul et même médecin,
    # deux créneaux différents selon le jour. Voir section 6bis pour le forçage.
    rythmo_vendredi_doctor = "P" if req.week_type == 1 else "U"
    rythmo_vendredi_slot = "matin" if req.week_type == 1 else "am"
    NCT_ALLOWED = set(rules["nct_allowed"])
    NCT_FIXED_SCHEDULE = rules["nct_fixed_schedule"]
    GARDE_ALLOWED = set(rules.get("garde_allowed", []))  # vide = pas de restriction (rétro-compatible)

    # --- 1. Préparation des données ---
    medecins_map = {m.id: m for m in req.medecins}
    astreinte_coro_ids = {m.id for m in req.medecins if m.statut == StatutMedecin.ASTREINTE_CORO}  # W, O, M
    wom_pool = rules["wom_pool"]
    nct_pool = {m.id for m in req.medecins if m.statut == StatutMedecin.ASTREINTE_CORO and m.id != "O"}
    fv_id = next((m.id for m in req.medecins if m.statut == StatutMedecin.FV), None)
    daas_id = next((m.id for m in req.medecins if m.statut == StatutMedecin.DAAS), None)
    d_id = next((m.id for m in req.medecins if m.statut == StatutMedecin.D), None)

    # Demi-journées libres et exclusions fixes : chargées depuis rules_config.json
    half_days_off = half_days_off_as_dict(rules)
    fixed_exclusions = fixed_exclusions_as_dict(rules)

    # --- Règle dynamique : 1/2 journée off après garde de nuit ---
    # (confirmée avec l'utilisateur : par défaut l'après-midi du lendemain ; le matin
    # si l'après-midi du lendemain est déjà un off fixe pour ce médecin ce jour-là -
    # pour éviter de "gaspiller" le repos sur un créneau déjà libre par ailleurs).
    # Calcul statique (ne dépend que de la config, pas de la solution) : pour chaque
    # (médecin, jour où il pourrait faire une garde de nuit), quel créneau cible le
    # lendemain.
    def target_off_slot_after_night_guard(doc_id: str, next_day_name: str) -> str:
        if (next_day_name, "am") in half_days_off and doc_id in half_days_off[(next_day_name, "am")]:
            return "matin"
        return "am"

    # --- 2. Création des variables ---
    model = cp_model.CpModel()
    x = {}  # (doc, day_idx, slot, activity) -> BoolVar

    def _is_rythmo_day(doc_id: str, day_name: str) -> bool:
        if day_name == "VENDREDI" and doc_id == rythmo_vendredi_doctor:
            return True
        return doc_id in RYTHMO_SCHEDULE and day_name in RYTHMO_SCHEDULE[doc_id]

    def _rythmo_slots_for(doc_id: str, day_name: str) -> Tuple[str, ...]:
        if day_name == "VENDREDI" and doc_id == rythmo_vendredi_doctor:
            return (rythmo_vendredi_slot,)
        return RYTHMO_SLOTS.get(doc_id, ("matin", "am"))

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
        if (day_name, slot) in half_days_off and doc_id in half_days_off[(day_name, slot)]:
            return

        # RYTHMO : exclusion des AUTRES activités uniquement sur le(s) créneau(x)
        # réellement occupé par RYTHMO (précision par médecin - P=matin+am,
        # U/A=am seul, vendredi=1 seul créneau en alternance). Les autres
        # créneaux du même jour restent disponibles pour d'autres activités.
        if _is_rythmo_day(doc_id, day_name) and slot in _rythmo_slots_for(doc_id, day_name) and activity != "RYTHMO":
            return

        # Empêche une contradiction avec la règle de repos après garde de nuit
        # (section 3bis) : si le LENDEMAIN a un créneau RYTHMO forcé pour ce
        # médecin QUI CHEVAUCHE le créneau cible du repos automatique, celui-ci
        # essaierait de bloquer un créneau que RYTHMO force par ailleurs à 1 -
        # contradiction menant à "Aucune solution trouvée". On évite le conflit
        # à la source, mais UNIQUEMENT si les créneaux se chevauchent réellement
        # (ex: A ne fait Rythmo que l'am, donc une garde nuit la veille reste OK
        # si le repos cible le matin).
        if slot == "nuit" and activity in ("GARDE", "ASTREINTE") and d_idx < 6:
            next_day_name = DAY_NAMES_FR[d_idx + 1]
            if _is_rythmo_day(doc_id, next_day_name):
                target_off = target_off_slot_after_night_guard(doc_id, next_day_name)
                if target_off in _rythmo_slots_for(doc_id, next_day_name):
                    return

        if doc_id in fixed_exclusions and d_idx in fixed_exclusions[doc_id]:
            return

        # Restrictions explicites par code médecin (priment sur les règles par statut)
        if activity == "REEDUC":
            if day_name not in REEDUC_DAYS or slot != "am" or doc_id not in REEDUC_ALLOWED:
                return
        if activity == "CORO" and doc_id not in CORO_ALLOWED:
            return
        if activity == "RYTHMO":
            if doc_id not in RYTHMO_ALLOWED:
                return
            # Un médecin du pool RYTHMO ne fait rythmo QUE son jour + créneau désignés
            # (ex: P ne fait jamais rythmo un autre jour que mardi, ni le matin
            # un jour où seul l'après-midi est prévu).
            if not _is_rythmo_day(doc_id, day_name):
                return
            if slot not in _rythmo_slots_for(doc_id, day_name):
                return
        if activity == "NCT" and (doc_id not in NCT_ALLOWED or d_idx != 3):
            return
        # Restriction demandée : la garde n'est répartie qu'entre les médecins listés
        # dans rules_config.json (garde_allowed). Ne s'applique pas à FV, dont la garde
        # fixe du lundi est déjà gérée par la règle spécifique ci-dessus.
        if activity == "GARDE" and doc_id != fv_id and GARDE_ALLOWED and doc_id not in GARDE_ALLOWED:
            return

        statut = medecins_map[doc_id].statut
        if statut == StatutMedecin.CH:
            return
        if statut == StatutMedecin.PERMANENT:
            if activity not in ("ASTREINTE", "GARDE", "REEDUC", "PRE_OP", "RYTHMO"):
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

    # --- 2bis. Activités historiques (Cs, ETT, EE...) ---
    # L'éligibilité N'EST PAS une liste écrite à la main (REEDUC_ALLOWED etc.)
    # mais DÉDUITE de historical_patterns (construit côté front à partir de
    # l'historique réel - voir lib/pattern-analysis.ts). Un médecin n'apparaît
    # comme éligible que s'il a déjà occupé ce (row_key, jour) au moins une
    # fois par le passé. Les activités "Hors site" (2ter, juste après) ont
    # leurs propres règles fixes, distinctes de ce mécanisme.

    # --- 2ter. Activités hors site à règles fixes (IRM, CDL, Scinti, LFB, PSSL) ---
    # Contrairement à 2bis (Cs/ETT/EE, déduit de l'historique), celles-ci ont des
    # pools de médecins fixes confirmés par l'utilisateur - pas une inférence de
    # données. Si le front envoie aussi ces row_key dans historical_patterns (par
    # ex. via un ancien calcul générique), on les ignore ici volontairement pour
    # ne pas créer deux jeux de variables concurrents pour la même case.
    hors_site_vars: Dict[tuple, Any] = {}  # (row_key, d_idx) -> {doc_id: BoolVar}
    hors_site_priority_bonus = []
    full_day_hors_site_vars: List[tuple] = []  # (doc_id, d_idx, var) - appliqué après création de toutes les vars

    for row_key, config in HORS_SITE_CONFIG.items():
        allowed = config["allowed"]
        full_day = config["full_day"]
        priority = config.get("priority", [])
        fixed_slots = config.get("fixed_slots")
        activity_name = f"HORSSITE::{row_key}"

        if fixed_slots:
            # Créneaux fixes forcés (ex: IRM lundi matin + vendredi am, ne
            # changent jamais) - un seul médecin autorisé attendu, forcé ==1,
            # PAS un choix optionnel comme les autres activités hors site.
            for day_name, slot in fixed_slots:
                d_idx = DAY_NAMES_FR.index(day_name)
                for doc_id in allowed:
                    if doc_id not in medecins_map:
                        continue
                    if is_on_vacation(doc_id, days[d_idx], req.vacations) or is_on_vacation(doc_id, days[d_idx], req.congres):
                        warnings.append(
                            f"{row_key} : {doc_id} en congé le {day_name} - créneau fixe non couvert cette semaine."
                        )
                        continue
                    var = model.NewBoolVar(f"horssite_{doc_id}_{d_idx}_{row_key}")
                    x[(doc_id, d_idx, slot, activity_name)] = var
                    model.Add(var == 1)  # forcé, pas optionnel
                    if full_day:
                        full_day_hors_site_vars.append((doc_id, d_idx, var))
            continue

        for d_idx in range(7):
            day_vars: Dict[str, Any] = {}
            for doc_id in allowed:
                if doc_id not in medecins_map:
                    continue
                if is_on_vacation(doc_id, days[d_idx], req.vacations) or is_on_vacation(doc_id, days[d_idx], req.congres):
                    continue
                var = model.NewBoolVar(f"horssite_{doc_id}_{d_idx}_{row_key}")
                x[(doc_id, d_idx, "matin", activity_name)] = var
                day_vars[doc_id] = var

                # Priorité (ex: CDL, V > O) : bonus fort pour le 1er de la
                # liste `priority`, pour que le solveur le choisisse toujours
                # sauf s'il est indisponible/occupé ailleurs ce jour-là.
                if priority and doc_id in priority:
                    rank = priority.index(doc_id)
                    weight = 100 if rank == 0 else 1
                    hors_site_priority_bonus.append(weight * var)

                # Journée entière : collecté ici, appliqué APRÈS la création de
                # TOUTES les variables (historique, Entrées PSS...) - sinon la
                # réification ne capturerait que les vars "am" déjà existantes
                # à ce stade, ratant Cs/ETT/Stress/Entrées PSS créées plus loin.
                if full_day:
                    full_day_hors_site_vars.append((doc_id, d_idx, var))

            if day_vars:
                model.Add(sum(day_vars.values()) <= 1)
                hors_site_vars[(row_key, d_idx)] = day_vars

    historical_vars: Dict[tuple, Any] = {}  # (row_key, d_idx) -> {doc_id: (BoolVar, frequency)}
    historical_patterns = req.historical_patterns or {}

    for row_key, by_day in historical_patterns.items():
        if row_key in HORS_SITE_CONFIG or row_key == "Entrées PSS":
            # Entrées PSS a ses propres règles dédiées (voir 2quater) : roulement
            # lundi/mardi parmi le pool clinique, jumelé au garde de l'après-midi
            # mercredi/jeudi/vendredi.
            continue

        # ½-off / Visite / Rythmo / Coro / Garde / Astreinte / NCT... : hors
        # périmètre de la fidélité historique (gérées par des règles dures
        # ailleurs) - ignorées silencieusement, pas de warning "non reconnu"
        # pour ce qui est normal et attendu.
        if not is_solver_historical_row_key(row_key):
            continue

        slot = map_historical_row_key_to_slot(row_key)
        if slot is None:
            warnings.append(f"historical_patterns : row_key '{row_key}' non reconnu, ignoré.")
            continue

        activity_name = f"HIST::{row_key}"  # préfixe technique pour ne jamais collider avec ACTIVITIES

        for day_name, pattern in by_day.items():
            if day_name not in DAY_NAMES_FR:
                continue
            d_idx = DAY_NAMES_FR.index(day_name)
            eligible = pattern.get("eligible_doctors", [])
            frequency = pattern.get("frequency", {})

            day_vars: Dict[str, tuple] = {}
            for doc_id in eligible:
                if doc_id not in medecins_map:
                    continue  # médecin externe/inactif, pas dans la requête courante
                if is_on_vacation(doc_id, days[d_idx], req.vacations) or is_on_vacation(doc_id, days[d_idx], req.congres):
                    continue
                var = model.NewBoolVar(f"hist_{doc_id}_{d_idx}_{row_key}")
                x[(doc_id, d_idx, slot, activity_name)] = var
                day_vars[doc_id] = (var, frequency.get(doc_id, 0))

            if day_vars:
                # Au plus une personne par créneau (pas de double-réservation),
                # mais PAS obligatoire (<=1, pas ==1) : on ne force pas un
                # remplissage si personne d'éligible n'est disponible cette
                # semaine-là (vacances etc.), plutôt que de risquer une
                # infaisabilité du modèle entier.
                model.Add(sum(v for v, _ in day_vars.values()) <= 1)
                historical_vars[(row_key, d_idx)] = day_vars

    # --- 2quater. Entrées PSS (confirmé avec l'utilisateur le 26/07/2026) ---
    # Lundi/mardi : roulement parmi les médecins affectés ce jour-là en Apm -
    # ETT salle 1/2, Apm - Stress, Apm - Cs PSS ou Apm - Cs Tessée (le "pool
    # clinique" de l'après-midi), à l'EXCLUSION de tout médecin qui, ce même
    # jour (n'importe quel créneau), fait CORO, RYTHMO, ou une activité hors
    # site (CDL/IRM/Scinti/LFB/PSSL).
    # Mercredi/jeudi/vendredi : Entrées PSS = automatiquement le même médecin
    # que le cardiologue de garde de l'après-midi (Garde Midi) ce jour-là -
    # pas un choix séparé, un jumelage direct.
    ENTREES_PSS_CLINIC_ROW_KEYS = [
        "Apm - ETT salle 1", "Apm - ETT salle 2", "Apm - Stress",
        "Apm - Cs PSS", "Apm - Cs Tessée",
    ]
    entrees_pss_activity = "HIST::Entrées PSS"
    entrees_pss_fill_bonus: List[Any] = []

    for d_idx in (0, 1):  # LUNDI, MARDI - roulement pool clinique
        day_name = DAY_NAMES_FR[d_idx]
        clinic_vars_by_doc: Dict[str, List[Any]] = {}
        for (doc_id_x, dd, sl, act), var in list(x.items()):
            if dd != d_idx or not act.startswith("HIST::"):
                continue
            if act[len("HIST::"):] in ENTREES_PSS_CLINIC_ROW_KEYS:
                clinic_vars_by_doc.setdefault(doc_id_x, []).append(var)

        day_vars: Dict[str, Any] = {}
        for doc_id_x, vlist in clinic_vars_by_doc.items():
            in_clinic_pool = model.NewBoolVar(f"clinic_pool_{doc_id_x}_{d_idx}")
            model.AddMaxEquality(in_clinic_pool, vlist)

            var = model.NewBoolVar(f"entreespss_{doc_id_x}_{d_idx}")
            x[(doc_id_x, d_idx, "am", entrees_pss_activity)] = var
            model.Add(var <= in_clinic_pool)  # ne peut être choisi que si dans le pool clinique ce jour

            # Exclusion CORO/RYTHMO/hors site l'APRÈS-MIDI uniquement (corrigé
            # le 26/07/2026 - pas toute la journée : un médecin en RYTHMO/CORO/
            # hors site le matin seulement reste éligible à Entrées PSS l'am).
            exclusion_vars = [
                v for (d2, dd2, sl2, act2), v in x.items()
                if d2 == doc_id_x and dd2 == d_idx and sl2 == "am"
                and (act2.startswith("HORSSITE::") or act2 in ("CORO", "RYTHMO"))
            ]
            for ev in exclusion_vars:
                model.Add(var == 0).OnlyEnforceIf(ev)

            day_vars[doc_id_x] = var

        if day_vars:
            model.Add(sum(day_vars.values()) <= 1)
            # Forte préférence de remplissage (pas une obligation stricte, pour
            # éviter une infaisabilité si tous les candidats du pool clinique
            # sont par ailleurs exclus ce jour-là) : sans ce bonus, l'activité
            # étant purement optionnelle, le solveur la laisserait vide par
            # défaut faute de coût associé à ne pas la remplir.
            filled_var = model.NewBoolVar(f"entreespss_filled_{d_idx}")
            model.AddMaxEquality(filled_var, list(day_vars.values()))
            entrees_pss_fill_bonus.append(20 * filled_var)
            if not clinic_vars_by_doc:
                warnings.append(
                    f"Entrées PSS {day_name} : aucun médecin en Cs/ETT/Stress "
                    f"cet après-midi d'après l'historique reçu - rien à proposer."
                )

    for d_idx in (2, 3, 4):  # MERCREDI, JEUDI, VENDREDI - jumelé au garde de garde Midi
        day_name = DAY_NAMES_FR[d_idx]
        garde_midi_vars = {
            doc_id_x: v for (doc_id_x, dd, sl, act), v in x.items()
            if dd == d_idx and sl == "am" and act == "GARDE"
        }
        for doc_id_x, garde_var in garde_midi_vars.items():
            var = model.NewBoolVar(f"entreespss_{doc_id_x}_{d_idx}")
            x[(doc_id_x, d_idx, "am", entrees_pss_activity)] = var
            # Jumelage strict : Entrées PSS == Garde Midi (même médecin, pas de choix séparé)
            model.Add(var == garde_var)

    # --- Application différée du blocage "journée entière" (LFB, PSSL...) ---
    # Fait ICI, une fois TOUTES les variables créées (historique, Entrées PSS
    # inclus), pour ne rater aucune activité "am" apparue plus tard dans le code.
    for doc_id, d_idx, hors_site_var in full_day_hors_site_vars:
        am_vars_same_day = [
            v for (d, dd, sl, act), v in x.items()
            if d == doc_id and dd == d_idx and sl == "am" and v is not hors_site_var
        ]
        for av in am_vars_same_day:
            model.Add(av == 0).OnlyEnforceIf(hors_site_var)

    # --- 3. Contraintes générales ---
    # Règle absolue : jamais 2 médecins sur une même case (jour + créneau + activité),
    # applicable à toutes les activités gérées par le solveur : ASTREINTE, GARDE, CORO, NCT.
    # (Les catégories vacances / congé / congrès / 1/2 journée libre / ETT salle 1-2 ne sont
    # pas soumises à cette règle : elles sont traitées séparément, cf. section 13bis.)
    for d_idx in range(7):
        for slot in SLOTS:
            for activity in ACTIVITIES:
                case_vars = [v for (doc, d, sl, act), v in x.items() if d == d_idx and sl == slot and act == activity]
                if case_vars:
                    model.Add(sum(case_vars) <= 1)

    # Un médecin ne fait qu'une activité par créneau
    # (Entrées PSS exceptée : elle se SUPERPOSE intentionnellement à Cs/ETT/
    # Stress/Garde Midi ce même créneau - voir 2quater - ce n'est pas une
    # activité concurrente pour le temps du médecin, juste une case de plus
    # dans le planning affichant la même personne.)
    for doc_id in medecins_map:
        for d_idx in range(7):
            for slot in SLOTS:
                slot_vars = [
                    v for (doc, d, sl, act), v in x.items()
                    if d == d_idx and sl == slot and doc == doc_id and act != entrees_pss_activity
                ]
                if slot_vars:
                    model.Add(sum(slot_vars) <= 1)

    # --- 3bis. Règle dynamique : 1/2 journée off après garde de nuit ---
    # Si un médecin travaille en garde/astreinte de nuit le jour d, il ne peut avoir
    # aucune autre activité sur le créneau cible du jour d+1 (voir
    # target_off_slot_after_night_guard ci-dessus). Réifié avec OnlyEnforceIf : la
    # contrainte ne s'active QUE si la garde de nuit est effectivement retenue par le
    # solveur cette semaine-là (pas une exclusion a priori comme half_days_off).
    # Limité à d_idx 0..5 (lundi->samedi) : le cas dimanche->lundi traverse deux
    # semaines, géré séparément juste après via previous_sunday_guard_doctor.
    post_night_guard_off_flags: Dict[tuple, Any] = {}  # (doc_id, d_idx) -> BoolVar "a fait une garde de nuit ce jour-là"

    for doc_id in medecins_map:
        for d_idx in range(6):  # LUNDI(0) à SAMEDI(5)
            night_vars = [
                v for (doc, d, sl, act), v in x.items()
                if doc == doc_id and d == d_idx and sl == "nuit" and act in ("GARDE", "ASTREINTE")
            ]
            if not night_vars:
                continue

            worked_night = model.NewBoolVar(f"worked_night_{doc_id}_{d_idx}")
            model.Add(sum(night_vars) >= 1).OnlyEnforceIf(worked_night)
            model.Add(sum(night_vars) == 0).OnlyEnforceIf(worked_night.Not())
            post_night_guard_off_flags[(doc_id, d_idx)] = worked_night

            next_day_name = DAY_NAMES_FR[d_idx + 1]
            target_slot = target_off_slot_after_night_guard(doc_id, next_day_name)

            other_vars_next_day = [
                v for (doc, d, sl, act), v in x.items()
                if doc == doc_id and d == d_idx + 1 and sl == target_slot
            ]
            for v in other_vars_next_day:
                model.Add(v == 0).OnlyEnforceIf(worked_night)

    # Cas dimanche (semaine précédente) -> lundi (cette semaine) : le doctor est connu
    # à l'avance (transmis par le front), donc traité comme une exclusion fixe
    # classique plutôt qu'une réification - pas besoin de deviner, on SAIT déjà que
    # ce médecin a fait la garde/astreinte dimanche dernier.
    if req.previous_sunday_guard_doctor:
        sunday_doc = req.previous_sunday_guard_doctor
        monday_name = DAY_NAMES_FR[0]
        # Si RYTHMO est forcé sur le créneau de repos ciblé lundi, cette règle
        # ne s'applique pas (RYTHMO prime) - mais uniquement en cas de
        # chevauchement réel de créneau (ex: A ne fait Rythmo que l'am, donc
        # un repos matin reste appliqué normalement).
        target_slot = target_off_slot_after_night_guard(sunday_doc, monday_name)
        rythmo_blocks_off = _is_rythmo_day(sunday_doc, monday_name) and target_slot in _rythmo_slots_for(sunday_doc, monday_name)
        if not rythmo_blocks_off:
            for (doc, d, sl, act), v in x.items():
                if doc == sunday_doc and d == 0 and sl == target_slot:
                    model.Add(v == 0)
        else:
            warnings.append(
                f"{sunday_doc} a fait la garde de nuit dimanche dernier mais est en RYTHMO "
                f"forcé ce lundi ({target_slot}) - repos automatique non appliqué (RYTHMO prime)."
            )

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
    thursday_iso = days[3].isoformat()
    nct_vars = [v for (doc, d, sl, act), v in x.items() if d == 3 and sl == "nuit" and act == "NCT"]

    if thursday_iso in NCT_FIXED_SCHEDULE:
        # Calendrier déjà planifié à l'avance : prime sur l'alternance/l'équité pour ce jeudi précis.
        fixed_doctor = NCT_FIXED_SCHEDULE[thursday_iso]
        var_fixed = x.get((fixed_doctor, 3, "nuit", "NCT"))
        if var_fixed is not None:
            model.Add(var_fixed == 1)
            for (doc, d, sl, act), var in x.items():
                if d == 3 and sl == "nuit" and act == "NCT" and doc != fixed_doctor:
                    model.Add(var == 0)
        else:
            warnings.append(
                f"JEUDI {thursday_iso} : NCT planifiée pour {fixed_doctor} mais ce médecin "
                f"n'est pas disponible ce jour (vacances ou exclu) - à réassigner manuellement"
            )
    else:
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

    # --- 5bis. REEDUC (obligatoire, 1 médecin exactement, Lundi/Mercredi/Vendredi am) ---
    for d_idx, day_nm in enumerate(DAY_NAMES_FR):
        if day_nm not in REEDUC_DAYS:
            continue
        reeduc_vars = [v for (doc, d, sl, act), v in x.items() if d == d_idx and sl == "am" and act == "REEDUC"]
        if reeduc_vars:
            model.Add(sum(reeduc_vars) == 1)
        else:
            warnings.append(f"{day_nm} : aucun médecin disponible pour REEDUC (vacances ou exclu)")

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

    # --- 6bis. RYTHMO forcé sur le(s) créneau(x) exact(s) par jour/médecin ---
    # A = après-midi seulement (lundi, jeudi) ; P = matin+après-midi (mardi) ;
    # U = matin+après-midi (mercredi) ; vendredi = alternance P/U selon
    # week_type, un seul créneau (voir rythmo_schedule/rythmo_slots dans
    # rules_config.json + rythmo_vendredi_doctor/slot ci-dessus). Confirmé
    # précisément avec l'utilisateur le 27/07/2026.
    def _force_rythmo(doc_id: str, day_name: str, slot: str):
        d_idx = DAY_NAMES_FR.index(day_name)
        var = x.get((doc_id, d_idx, slot, "RYTHMO"))
        if var is not None:
            model.Add(var == 1)
        else:
            warnings.append(
                f"RYTHMO {doc_id} non disponible {day_name} {slot} (vacances/congé ce jour-là ?)"
            )

    for doc_id, rythmo_days in RYTHMO_SCHEDULE.items():
        for day_name in rythmo_days:
            for slot in RYTHMO_SLOTS.get(doc_id, ("matin", "am")):
                _force_rythmo(doc_id, day_name, slot)

    _force_rythmo(rythmo_vendredi_doctor, "VENDREDI", rythmo_vendredi_slot)

    # --- 7. Règles d'exclusion métier ---
    # 7.1 AM OFF après garde nuit (lendemain matin)
    for doc_id in medecins_map:
        for d_idx in range(6):
            var_nuit_garde = x.get((doc_id, d_idx, "nuit", "GARDE"))
            if var_nuit_garde is None:
                continue
            am_next_vars = [v for (doc, d, sl, act), v in x.items() if d == d_idx + 1 and sl == "matin" and doc == doc_id]
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
            am_vars = [v for (doc, d, sl, act), v in x.items() if d == d_idx and sl == "am" and doc == doc_id]
            if am_vars:
                presence_am = model.NewBoolVar(f"presence_am_{doc_id}_{d_idx}")
                model.Add(sum(am_vars) >= 1).OnlyEnforceIf(presence_am)
                model.Add(sum(am_vars) == 0).OnlyEnforceIf(presence_am.Not())
                model.AddImplication(var_nuit_garde, presence_am.Not())

    # 7.3 Pas d'astreinte nuit si garde ce jour
    for doc_id in medecins_map:
        for d_idx in range(7):
            garde_vars = [v for (doc, d, sl, act), v in x.items() if d == d_idx and doc == doc_id and act == "GARDE"]
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
        for combined_key, doctors in req.existing_schedule.items():
            row_key, _, day_name = combined_key.partition("||")
            slot, activity = map_row_key_to_slot_activity(row_key)
            if slot is None or activity is None:
                continue
            day_idx = DAY_NAMES_FR.index(day_name)
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
    # Poids relatifs par type d'activité (harmonisés avec les compteurs historiques
    # points_astreinte / points_garde / points_nct / points_weekend envoyés par le front).
    # Configurable ici sans toucher au reste du solveur.
    WEIGHT_ASTREINTE = 1
    WEIGHT_GARDE = 1
    WEIGHT_NCT = 2       # nuit du jeudi, plus contraignante
    WEIGHT_WEEKEND = 2   # astreinte de weekend, plus contraignante

    def activity_weight(activity: str, d_idx: int) -> int:
        if activity == "NCT":
            return WEIGHT_NCT
        if activity == "ASTREINTE" and d_idx in (5, 6):
            return WEIGHT_WEEKEND
        if activity == "ASTREINTE":
            return WEIGHT_ASTREINTE
        if activity == "GARDE":
            return WEIGHT_GARDE
        return 0  # CORO / REEDUC / RYTHMO / PRE_OP : hors périmètre équité garde/astreinte

    # Médecins concernés par l'équité garde/astreinte : tous les statuts qui peuvent
    # réellement être affectés à ASTREINTE/GARDE/NCT (PERMANENT + ASTREINTE_CORO).
    # Exclus : CH (structure fixe), FV/DAAS/D (créneaux externes fixes), ADMIN.
    equity_doctor_ids = {
        m.id for m in req.medecins
        if m.statut in (StatutMedecin.PERMANENT, StatutMedecin.ASTREINTE_CORO)
    }

    total_points: Dict[str, Any] = {}
    for doc in equity_doctor_ids:
        m = medecins_map[doc]
        pct = m.poids_equite_pct if m.poids_equite_pct > 0 else 100
        # Facteur de normalisation : un médecin à pct=50 (mi-temps, ou charge cible
        # réduite pour ancienneté) voit chacun de ses points compter double dans le
        # calcul d'équité normalisé -> le solveur cherchera naturellement à lui donner
        # moins de gardes/astreintes en valeur brute pour atteindre le même niveau
        # "normalisé" que ses collègues à pct=100.
        norm = lambda w: round(w * 100 / pct)  # noqa: E731

        historical = (
            m.points_astreinte * norm(WEIGHT_ASTREINTE)
            + m.points_garde * norm(WEIGHT_GARDE)
            + m.points_nct * norm(WEIGHT_NCT)
            + m.points_weekend * norm(WEIGHT_WEEKEND)
        )
        this_week_terms = [
            norm(activity_weight(activity, d_idx)) * var
            for (doc_id_x, d_idx, slot, activity), var in x.items()
            if doc_id_x == doc and activity_weight(activity, d_idx) > 0
        ]
        # IntVar bornée large pour couvrir historique + charge de la semaine
        # (le facteur de normalisation peut monter la borne pour les petites quotités).
        upper_bound = historical + 7 * norm(WEIGHT_NCT + WEIGHT_WEEKEND + WEIGHT_GARDE) + 10
        total_var = model.NewIntVar(0, max(upper_bound, 1), f"total_points_{doc}")
        model.Add(total_var == historical + sum(this_week_terms))
        total_points[doc] = total_var

    if total_points:
        max_points = model.NewIntVar(0, 100000, "max_points")
        min_points = model.NewIntVar(0, 100000, "min_points")
        for doc, var in total_points.items():
            model.Add(max_points >= var)
            model.Add(min_points <= var)
    else:
        warnings.append("Équité : aucun médecin éligible trouvé pour l'objectif d'équité")
        max_points = model.NewConstant(0)
        min_points = model.NewConstant(0)

    # --- Équité CORO (M, O, W uniquement - seuls coronarographistes internes,
    #     FV en fait aussi mais est externe et hors périmètre d'équité) ---
    # Scope MENSUEL et non cumulé à vie : `m.points_coro` est déjà calculé côté
    # front pour ne représenter QUE le mois calendaire en cours (voir
    # getMonthlyCoroEquity() dans app/actions/guard-api-actions.ts), contrairement
    # à points_astreinte/garde/nct/weekend qui sont cumulés depuis toujours.
    WEIGHT_CORO = 1
    coro_points: Dict[str, Any] = {}
    for doc in astreinte_coro_ids:
        m = medecins_map[doc]
        historical_coro = m.points_coro * WEIGHT_CORO
        this_week_coro = [
            WEIGHT_CORO * var
            for (doc_id_x, d_idx, slot, activity), var in x.items()
            if doc_id_x == doc and activity == "CORO"
        ]
        upper_bound = historical_coro + 7 * WEIGHT_CORO + 5
        coro_var = model.NewIntVar(0, max(upper_bound, 1), f"coro_points_{doc}")
        model.Add(coro_var == historical_coro + sum(this_week_coro))
        coro_points[doc] = coro_var

    # --- Bonus de fidélité historique (Cs/ETT/EE/hors site) ---
    # Récompense (dans l'objectif, donc soustrait puisqu'on minimise) le fait
    # d'assigner le médecin historiquement le plus fréquent sur ce créneau.
    # Poids volontairement modeste : ne doit pas dominer l'équité garde/astreinte
    # (l'enjeu principal), juste départager entre plusieurs solutions
    # équivalentes côté équité en faveur de ce qui ressemble le plus au passé.
    WEIGHT_HISTORICAL_FIDELITY = 1
    historical_bonus_terms = [
        WEIGHT_HISTORICAL_FIDELITY * freq * var
        for day_vars in historical_vars.values()
        for (var, freq) in day_vars.values()
        if freq > 0
    ]
    historical_bonus = sum(historical_bonus_terms) if historical_bonus_terms else 0
    hors_site_bonus = sum(hors_site_priority_bonus) if hors_site_priority_bonus else 0
    entrees_pss_bonus = sum(entrees_pss_fill_bonus) if entrees_pss_fill_bonus else 0

    if coro_points:
        coro_max = model.NewIntVar(0, 100000, "coro_max")
        coro_min = model.NewIntVar(0, 100000, "coro_min")
        for doc, var in coro_points.items():
            model.Add(coro_max >= var)
            model.Add(coro_min <= var)
        # Un seul model.Minimize() possible avec CP-SAT : on combine l'équité
        # garde/astreinte (poids fort, c'est l'enjeu principal), l'équité CORO
        # mensuelle (poids plus léger, sur un périmètre de 3 personnes seulement),
        # le bonus de fidélité historique (Cs/ETT/EE), la priorité hors site
        # (ex: CDL, H > O) et la préférence de remplissage Entrées PSS en une
        # seule expression additive.
        model.Minimize(
            (max_points - min_points) + (coro_max - coro_min)
            - historical_bonus - hors_site_bonus - entrees_pss_bonus
        )
    else:
        model.Minimize((max_points - min_points) - historical_bonus - hors_site_bonus - entrees_pss_bonus)


    # --- 12. Résolution ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)

    # --- 13. Extraction des résultats ---
    assignments: List[Assignment] = []

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for (doc, d_idx, slot, activity), var in x.items():
            if solver.Value(var) == 1:
                is_historical = activity.startswith("HIST::")
                is_hors_site = activity.startswith("HORSSITE::")
                if is_historical:
                    row_key = activity[len("HIST::"):]
                    clean_activity = row_key.split(" - ", 1)[1] if " - " in row_key else row_key
                elif is_hors_site:
                    row_key = activity[len("HORSSITE::"):]
                    clean_activity = row_key.split(" - ", 1)[1] if " - " in row_key else row_key
                else:
                    clean_activity = activity
                assignments.append(Assignment(
                    date=days[d_idx].isoformat(),
                    day_name=DAY_NAMES_FR[d_idx],
                    slot=slot,
                    activity=clean_activity,
                    doctor=doc,
                    note="assigné par le solveur (historique)" if is_historical else (
                        "assigné par le solveur (hors site)" if is_hors_site else "assigné par le solveur"
                    )
                ))

        # --- Repos dynamique après garde de nuit (voir section 3bis) ---
        for (doc_id, d_idx), worked_var in post_night_guard_off_flags.items():
            if solver.Value(worked_var) == 1:
                next_day_name = DAY_NAMES_FR[d_idx + 1]
                target_slot = target_off_slot_after_night_guard(doc_id, next_day_name)
                assignments.append(Assignment(
                    date=days[d_idx + 1].isoformat(),
                    day_name=next_day_name,
                    slot=target_slot,
                    activity="DEMI_JOURNEE_LIBRE",
                    doctor=doc_id,
                    note="Repos après garde de nuit"
                ))

        if req.previous_sunday_guard_doctor:
            sunday_doc = req.previous_sunday_guard_doctor
            monday_name = DAY_NAMES_FR[0]
            target_slot = target_off_slot_after_night_guard(sunday_doc, monday_name)
            rythmo_takes_priority = (
                _is_rythmo_day(sunday_doc, monday_name) and target_slot in _rythmo_slots_for(sunday_doc, monday_name)
            )
            if not rythmo_takes_priority:
                assignments.append(Assignment(
                    date=days[0].isoformat(),
                    day_name=monday_name,
                    slot=target_slot,
                    activity="DEMI_JOURNEE_LIBRE",
                    doctor=sunday_doc,
                    note="Repos après garde de nuit (dimanche précédent)"
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

    # --- 13bis. Lignes dérivées (non optimisées par le solveur) ---
    # Vacances : reflète directement req.vacations pour les jours de la semaine en cours.
    # Congé : contenu systématiquement identique à Vacances (retranscription automatique).
    for v in req.vacations:
        v_start = date.fromisoformat(v.start_date)
        v_end = date.fromisoformat(v.end_date)
        for d_idx, day in enumerate(days):
            if v_start <= day <= v_end:
                assignments.append(Assignment(
                    date=day.isoformat(), day_name=DAY_NAMES_FR[d_idx],
                    slot="weekend" if d_idx >= 5 else "matin", activity="VACANCES",
                    doctor=v.doctor_id, note="Saisie vacances"
                ))
                assignments.append(Assignment(
                    date=day.isoformat(), day_name=DAY_NAMES_FR[d_idx],
                    slot="weekend" if d_idx >= 5 else "matin", activity="CONGE",
                    doctor=v.doctor_id, note="Retranscrit automatiquement depuis Vacances"
                ))

    # Congrès : même logique que vacances, à partir de req.congres.
    for c in req.congres:
        c_start = date.fromisoformat(c.start_date)
        c_end = date.fromisoformat(c.end_date)
        for d_idx, day in enumerate(days):
            if c_start <= day <= c_end:
                assignments.append(Assignment(
                    date=day.isoformat(), day_name=DAY_NAMES_FR[d_idx],
                    slot="weekend" if d_idx >= 5 else "matin", activity="CONGRES",
                    doctor=c.doctor_id, note="Saisie congrès"
                ))

    # 1/2 journée libre : dérivée directement de la règle métier déjà codée (half_days_off).
    for (day_name_key, slot_key), doctors_off in half_days_off.items():
        d_idx = DAY_NAMES_FR.index(day_name_key)
        for doc in doctors_off:
            if doc in medecins_map:
                assignments.append(Assignment(
                    date=days[d_idx].isoformat(), day_name=day_name_key,
                    slot=slot_key, activity="DEMI_JOURNEE_LIBRE",
                    doctor=doc, note="Règle fixe demi-journée libre"
                ))

    # Trier
    assignments.sort(key=lambda a: (a.date, SLOTS.index(a.slot) if a.slot in SLOTS else 999))

    return GenerateWeekResponse(
        week_start_date=req.week_start_date,
        assignments=assignments,
        warnings=warnings
    )
