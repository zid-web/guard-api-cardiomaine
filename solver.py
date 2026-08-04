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
    points_coro: int = 0  # Équité CORO : scope 6 MOIS glissants, pertinent uniquement pour M/O/W.
    # Équité Groupe 1 (échographistes B,Z,H,G,S) - scope 6 MOIS glissants,
    # trois métriques SÉPARÉES (pas combinées) : même nombre de Cs, même
    # nombre d'ETT, même nombre de Stress - confirmé utilisateur 28/07/2026.
    points_cs: int = 0
    points_ett: int = 0
    points_stress: int = 0
    points_ee: int = 0
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

class RoomMaintenance(BaseModel):
    """Salle de coronarographie indisponible sur une période - bloque CORO
    pour TOUS les médecins concernés (M, O, W, FV), pas un médecin en
    particulier, mais UNIQUEMENT sur le(s) créneau(x) réellement en
    maintenance (ex: après-midi seul, pas forcément matin+après-midi)."""
    start_date: str
    end_date: str
    slots: List[str] = ["matin", "am"]  # sous-ensemble concerné, ex: ["am"] seul
    reason: Optional[str] = None  # ex: "maintenance", informationnel uniquement

class ActivityMaintenance(BaseModel):
    """Suspension d'une activité entière sur une période (ex: NCT suspendue
    S31-S36, hors site PSSL/LFB/CDL suspendus S28-S36) - bloque TOUT LE MONDE,
    pas un médecin en particulier. Distinct de RoomMaintenance (dédié à Coro)
    pour rester simple : une période + une liste d'activités concernées."""
    start_date: str
    end_date: str
    activities: List[str]  # ex: ["NCT"] ou ["PSSL", "LFB", "CDL"]
    reason: Optional[str] = None

class PartialAbsence(BaseModel):
    """Absence ponctuelle d'un médecin sur un/des créneau(x) précis d'une seule
    journée - granularité plus fine qu'une Vacation (qui bloque la journée
    entière). Ex: S absent jeudi matin seulement."""
    doctor_id: str
    date: str  # YYYY-MM-DD, un seul jour (pas de plage - "ponctuelle")
    slots: List[str]  # sous-ensemble de ["matin", "am", "nuit"]

class GenerateWeekRequest(BaseModel):
    week_start_date: str              # YYYY-MM-DD (lundi)
    week_type: int                    # 1 = impaire, 2 = paire
    medecins: List[Medecin]
    vacations: List[Vacation] = []
    congres: List[Vacation] = []      # même structure que vacations : doctor_id, start_date, end_date
    room_maintenance: List[RoomMaintenance] = []  # salle de coro indisponible sur une période
    activity_maintenance: List[ActivityMaintenance] = []  # NCT/PSSL/LFB/CDL suspendus sur une période (confirmé utilisateur 29/07/2026)
    partial_absences: List[PartialAbsence] = []   # absence ponctuelle par créneau précis (voir modèle ci-dessus)
    weekend_mode: Literal["CH", "ROTATION"] = "ROTATION"
    last_nct_doctor: Optional[str] = None  # W ou M
    previous_sunday_guard_doctor: Optional[str] = None  # médecin ayant fait la garde/astreinte
                                                          # de nuit dimanche (semaine précédente) ;
                                                          # sert à forcer son 1/2 journée off ce lundi
                                                          # (règle métier: "Garde de nuit dimanche ->
                                                          # 1/2 journée off lundi, appliquée systématiquement")
    visite_doctor: Optional[str] = None  # A, B ou U : qui est en semaine de VISITE cette semaine
                                          # (roulement 1 semaine sur 3, désigné en entrée - pas calculé
                                          # par le solveur, voir discussion utilisateur 28/07/2026 : la
                                          # rotation à 3 semaines ne s'aligne pas mécaniquement avec la
                                          # parité paire/impaire, un ajustement humain est nécessaire).
                                          # Conséquence : pas de Cs le matin cette semaine pour ce médecin.
    lfb_doctor: Optional[str] = None  # H, S ou G : qui fait LFB ce jeudi (roulement 1/3, désigné en
                                       # entrée comme visite_doctor - même raisonnement).
    pssl_doctor: Optional[str] = None  # B ou Z : qui fait PSSL ce jeudi (roulement 1/2, désigné en
                                        # entrée - remplace pssl_b_active/pssl_z_active, confirmé
                                        # utilisateur 29/07/2026 : B et Z alternent ENSEMBLE le jeudi,
                                        # pas séparément sur 2 jours distincts comme précédemment codé).
    fv_monday_night_active: bool = False  # FV fait la garde de nuit CE lundi précis (2 lundis par
                                           # mois seulement, pas systématique - confirmé utilisateur
                                           # 29/07/2026, remplace le forçage inconditionnel précédent).
                                           # Si False : S ou U peuvent prendre le relais (déjà dans
                                           # GARDE_ALLOWED, pas de liste de repli séparée nécessaire).
    weekend_astreinte_combo: bool = False  # Ce weekend (semaine WOM uniquement) est un weekend
                                            # "combo" garde+astreinte entre 2 des 3 coronarographistes
                                            # (~10 weekends/6 mois, désignés en entrée - confirmé
                                            # utilisateur 30/07/2026, même raisonnement que VISITE).
    weekend_combo_astreinte_anchor: Optional[str] = None  # M, O ou W : PRÉFÉRENCE pour faire astreinte
                                                            # (ven nuit + sam matin/midi/nuit) + garde
                                                            # dimanche - souple (confirmé utilisateur
                                                            # 30/07/2026) : si absent(e)/en formation ce
                                                            # weekend, le solveur ajuste automatiquement
                                                            # parmi les 2 autres membres du pool WOM.
    weekend_combo_garde_anchor: Optional[str] = None  # M, O ou W : PRÉFÉRENCE pour faire garde samedi +
                                                        # astreinte dimanche - même souplesse que ci-dessus.
    last_combo_garde_doctor: Optional[str] = None  # Dernier médecin ayant fait le rôle "garde" d'un
                                                     # weekend combo (M, O ou W) - pour éviter 2 fois le
                                                     # même à moins de 15 jours d'intervalle (confirmé
                                                     # utilisateur 30/07/2026, même logique que last_nct_doctor).
    last_combo_garde_date: Optional[str] = None  # Date (YYYY-MM-DD) du dernier weekend combo effectué
                                                   # par last_combo_garde_doctor - sert à calculer l'écart
                                                   # de 15 jours.
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

# Infirmières (Val, Véro, Laura) - binôme obligatoire avec un médecin sur
# Stress/EE (confirmé utilisateur 31/07/2026, même pools que côté front
# lib/nurse-rules.ts). Le solveur ne gère PAS le placement de l'infirmière
# elle-même (positionnée par le front via existing_schedule) - seulement la
# proposition du médecin partenaire pour les cases où elle est déjà présente.
NURSES = {"Val", "Véro", "Laura"}
STRESS_PARTNER_POOL = ["Z", "B", "D", "H", "G", "S", "K"]
EE_PARTNER_POOL = ["Z", "B", "D", "K", "R", "O", "P", "U", "A", "M", "W", "V", "H", "S", "G"]
STRESS_ROWS = {"Matin - Stress", "Apm - Stress"}
EE_ROWS = {"Matin - EE1", "Apm - EE1", "Matin - EE2", "Apm - EE2"}

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


def is_room_under_maintenance(day: date, slot: str, room_maintenance: List["RoomMaintenance"]) -> bool:
    """Salle de coro indisponible ce jour-là ET ce créneau précis (matin et
    après-midi peuvent être affectés indépendamment) - bloque CORO pour tout
    le monde sur ce seul créneau."""
    for m in room_maintenance:
        start = date.fromisoformat(m.start_date)
        end = date.fromisoformat(m.end_date)
        if start <= day <= end and slot in m.slots:
            return True
    return False


def is_activity_suspended(day: date, activity_or_row: str, activity_maintenance: List["ActivityMaintenance"]) -> bool:
    """Activité entière suspendue ce jour-là (NCT, ou hors site PSSL/LFB/CDL) -
    bloque tout le monde, indépendamment du médecin. `activity_or_row` peut
    être le nom d'activité solveur (ex: "NCT") ou le nom de la ligne hors
    site sans préfixe (ex: "PSSL", "LFB", "CDL")."""
    for m in activity_maintenance:
        start = date.fromisoformat(m.start_date)
        end = date.fromisoformat(m.end_date)
        if start <= day <= end and activity_or_row in m.activities:
            return True
    return False


def is_partially_absent(doctor_id: str, day: date, slot: str, partial_absences: List["PartialAbsence"]) -> bool:
    """Absence ponctuelle sur un créneau précis (granularité plus fine qu'une
    Vacation) - ex: S absent jeudi matin seulement, disponible l'après-midi."""
    day_iso = day.isoformat()
    for pa in partial_absences:
        if pa.doctor_id == doctor_id and pa.date == day_iso and slot in pa.slots:
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
    # IRM : créneaux fixes (lundi matin, vendredi après-midi), S uniquement,
    # non-exclusif (S peut cumuler une autre activité sauf Cs/ETT/Stress).
    "Hors site - IRM":    {"allowed": ["S"], "full_day": False,
                            "fixed_slots_by_doctor": {"S": [("LUNDI", "matin"), ("VENDREDI", "am")]},
                            "non_exclusive": True},
    # Scinti : roulement R (mardi matin) / T (lundi+mercredi matin) reste une
    # PROPOSITION optionnelle, pas une contrainte fixe (confirmé utilisateur
    # 29/07/2026 - assoupli pour donner plus de marge au solveur, évite les
    # "créneau fixe non couvert" quand R/T indisponibles).
    "Hors site - Scinti": {"allowed": ["R", "T"], "full_day": False,
                            "preferred_slots_by_doctor": {
                                "R": [("MARDI", "matin")],
                                "T": [("LUNDI", "matin"), ("MERCREDI", "matin")],
                            }},
    # CDL : mardi matin uniquement (confirmé utilisateur 28/07/2026, remplace
    # "n'importe quel jour"), V prioritaire, O en repli si V indisponible.
    "Hors site - CDL":    {"allowed": ["V", "O"], "full_day": False,
                            "fixed_slots_by_doctor": {
                                "V": [("MARDI", "matin")],
                                "O": [("MARDI", "matin")],
                            },
                            "priority": ["V", "O"]},
    # LFB et PSSL : rotations à référence externe (1 semaine sur 3 / 1 jeudi
    # sur 3 / 1 mardi sur 2) - PAS gérées ici, désignées en entrée comme
    # VISITE (voir lfb_doctor / pssl_b_active / pssl_z_active plus bas dans
    # generate_week, confirmé utilisateur 28/07/2026 : même logique que
    # VISITE, une rotation à 3 ne s'aligne pas mécaniquement, ajustement
    # humain nécessaire).
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
    REEDUC_ALLOWED_EXTRA_BY_DAY: Dict[str, set] = {
        day: set(docs) for day, docs in rules.get("reeduc_allowed_extra_by_day", {}).items()
    }
    REEDUC_DAYS = set(rules["reeduc_days"])
    CORO_ALLOWED = set(rules["coro_allowed"])
    RYTHMO_ALLOWED = set(rules["rythmo_allowed"])
    # Rythmo : calendrier confirmé (utilisateur + DOC022, 28/07/2026).
    # Impaire : A Lun+Jeu am ; P Mar matin+am ; U Mer am + Ven am (fixe, pas d'alternance)
    # Paire   : A Lun+Jeu am ; P Mar matin+am ; U Mer matin+am ;
    #           Ven am = P systématiquement
    # Vendredi impaire : U normalement matin, MAIS après-midi si cette semaine
    # coïncide avec la semaine de visite de U (confirmé utilisateur 29/07/2026,
    # dépend donc de req.visite_doctor).
    if req.week_type == 1:
        ven_slot_u = "am" if req.visite_doctor == "U" else "matin"
        RYTHMO_FORCE = [
            ("A", "LUNDI", "am"), ("A", "JEUDI", "am"),
            ("P", "MARDI", "matin"), ("P", "MARDI", "am"),
            ("U", "MERCREDI", "am"), ("U", "VENDREDI", ven_slot_u),
        ]
    else:
        # Semaine paire, U de visite : conflit mercredi matin (U doit être en
        # visite ce matin-là, pas en rythmo) - repli confirmé utilisateur
        # 29/07/2026 : A prend le rythmo du mercredi MATIN à sa place, U
        # garde uniquement l'après-midi ce mercredi-là.
        mercredi_matin_doctor = "A" if req.visite_doctor == "U" else "U"
        RYTHMO_FORCE = [
            ("A", "LUNDI", "am"), ("A", "JEUDI", "am"),
            ("P", "MARDI", "matin"), ("P", "MARDI", "am"),
            (mercredi_matin_doctor, "MERCREDI", "matin"), ("U", "MERCREDI", "am"),
            ("P", "VENDREDI", "am"),
        ]
    NCT_ALLOWED = set(rules["nct_allowed"])
    # Restriction Cs PSS vs Cs Tessée par médecin (confirmé utilisateur
    # 28/07/2026, exclusion stricte pour les 13 médecins concernés).
    CS_TYPE_ALLOWED: Dict[str, str] = rules.get("cs_type_allowed", {})
    CS_PSS_ALLOWED = set(rules.get("cs_pss_allowed", ["A", "H", "Z", "M", "W", "O", "G", "P"]))
    CS_TESSEE_ALLOWED = set(rules.get("cs_tessee_allowed", ["B", "S", "U", "V", "T"]))
    ETT_ALLOWED = set(rules.get("ett_allowed", ["Z", "A", "R", "P", "M", "G", "S", "K", "H", "B", "Val"]))
    EE_ALLOWED = set(rules.get("ee_allowed", ["K", "V", "O", "W", "M", "Dass", "D", "R", "T", "H", "G", "U", "A"]))
    ETT_TESSEE_ALLOWED = set(rules.get("ett_tessee_allowed", ["Val"]))
    # ATL Matin/Midi/Soir = coronarographistes uniquement (M, O, W, FV, CH),
    # confirmé via DOC022 (28/07/2026) - PAS un pool large de PERMANENT.
    ASTREINTE_ALLOWED = set(rules.get("astreinte_allowed") or (list(CORO_ALLOWED) + ["CH"]))
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
        return any(d == doc_id and day == day_name for d, day, _slot in RYTHMO_FORCE)

    def _rythmo_slots_for(doc_id: str, day_name: str) -> Tuple[str, ...]:
        return tuple(slot for d, day, slot in RYTHMO_FORCE if d == doc_id and day == day_name)

    # Jours (pas encore les vars) des créneaux fixes Cs/ETT/doublon - utilisé
    # ci-dessous pour empêcher qu'un médecin soit affecté en garde de nuit un
    # jour où il a par ailleurs un engagement fixe l'après-midi (conflit avec
    # la règle 7.2 "garde nuit -> pas d'activité l'am ce jour-là" sinon,
    # confirmé bug réel le 29/07/2026 : S garde nuit mercredi + ETT ped
    # mercredi = infaisable).
    FIXED_AFTERNOON_COMMITMENT_DAYS = {
        ("S", "MERCREDI"), ("A", "MARDI"), ("P", "LUNDI"),  # FIXED_CS_ETT_SLOTS
        ("Z", "LUNDI"), ("H", "MARDI"),                      # DOUBLON_CS_CONFIG
    }

    def add_var_if_allowed(doc_id: str, d_idx: int, slot: str, activity: str):
        day = days[d_idx]
        if is_on_vacation(doc_id, day, req.vacations):
            return

        if is_partially_absent(doc_id, day, slot, req.partial_absences):
            return

        if activity == "CORO" and is_room_under_maintenance(day, slot, req.room_maintenance):
            return

        # Astreinte ATL suit automatiquement Coro (couplage confirmé DOC022,
        # 28/07/2026) - donc pendant une maintenance de la salle de coro,
        # l'astreinte des médecins coronarographistes doit AUSSI devenir
        # indisponible sur ce créneau, sinon le couplage ne peut jamais
        # s'appliquer (la variable Coro n'existe plus, rien à coupler) et
        # l'astreinte reste assignable à tort (confirmé bug utilisateur
        # 31/07/2026). Lundi-vendredi, matin/am uniquement - même périmètre
        # exact que le couplage ATL=Coro.
        if (
            activity == "ASTREINTE"
            and doc_id in CORO_ALLOWED
            and slot in ("matin", "am")
            and day.weekday() < 5
            and is_room_under_maintenance(day, slot, req.room_maintenance)
        ):
            return

        if doc_id in (daas_id, d_id):
            return

        if doc_id == fv_id:
            if not (d_idx == 0 and slot == "nuit" and activity == "GARDE" and req.fv_monday_night_active) and \
               not (d_idx == 3 and slot == "am" and activity == "CORO") and \
               not (activity == "ASTREINTE" and doc_id in ASTREINTE_ALLOWED):
                return

        # Exclusions de garde de nuit confirmées utilisateur 29/07/2026 :
        # O jamais mardi nuit ; M/O/W jamais vendredi nuit ; S jamais mardi nuit.
        if activity == "GARDE" and slot == "nuit":
            if doc_id == "O" and d_idx == 1:  # MARDI
                return
            if doc_id in ("M", "O", "W") and d_idx == 4:  # VENDREDI
                return
            if doc_id == "S" and d_idx == 1:  # MARDI (confirmé utilisateur 29/07/2026)
                return
            # Jamais garde de nuit le jour d'un engagement fixe l'après-midi
            # du même médecin (ETT ped, Cs PM, doublon) - éviterait sinon une
            # contradiction directe avec la règle 7.2.
            if (doc_id, DAY_NAMES_FR[d_idx]) in FIXED_AFTERNOON_COMMITMENT_DAYS:
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
            # Même logique pour les engagements fixes après-midi (ETT ped,
            # Cs PM, doublon) : le repos automatique du lendemain cible "am"
            # par défaut, ce qui chevauche directement ces créneaux fixes -
            # confirmé bug réel le 29/07/2026 (2e chemin, distinct du conflit
            # même-jour déjà corrigé).
            if (doc_id, next_day_name) in FIXED_AFTERNOON_COMMITMENT_DAYS:
                target_off = target_off_slot_after_night_guard(doc_id, next_day_name)
                if target_off == "am":
                    return

        if doc_id in fixed_exclusions and d_idx in fixed_exclusions[doc_id]:
            return

        # Restrictions explicites par code médecin (priment sur les règles par statut)
        if activity == "REEDUC":
            if day_name not in REEDUC_DAYS or slot != "am":
                return
            eligible_today = REEDUC_ALLOWED | REEDUC_ALLOWED_EXTRA_BY_DAY.get(day_name, set())
            if doc_id not in eligible_today:
                return
            # S est toujours occupé sur ETT ped le mercredi après-midi (créneau
            # fixe, voir FIXED_CS_ETT_SLOTS) - jamais éligible à REEDUC ce
            # jour-là, sous peine de contradiction directe si R et K sont
            # tous deux indisponibles (S resterait le seul candidat REEDUC
            # tout en étant déjà forcé sur ETT ped -> infaisabilité).
            if doc_id == "S" and day_name == "MERCREDI":
                return
        if activity == "CORO" and doc_id not in CORO_ALLOWED:
            return
        # ATL = coronarographistes uniquement (R/V/T/G... exclus). CH n'a pas
        # de BoolVar ici (statut CH -> return plus bas), injecté séparément.
        if activity == "ASTREINTE" and ASTREINTE_ALLOWED and doc_id not in ASTREINTE_ALLOWED:
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
        if activity == "NCT" and is_activity_suspended(day, "NCT", req.activity_maintenance):
            return
        # Restriction demandée : la garde n'est répartie qu'entre les médecins listés
        # dans rules_config.json (garde_allowed). Ne s'applique pas à FV, dont la garde
        # fixe du lundi est déjà gérée par la règle spécifique ci-dessus.
        if activity == "GARDE" and doc_id != fv_id and GARDE_ALLOWED and doc_id not in GARDE_ALLOWED:
            return
        # GARDE weekend (samedi/dimanche) : décidée lors d'une réunion tous les
        # 6 mois, pas par le solveur (confirmé utilisateur 29/07/2026) - SAUF
        # weekend "combo" désigné en entrée (confirmé utilisateur 30/07/2026),
        # où les 2 ancres (astreinte/garde) ont exactement une garde chacune
        # ce weekend (dimanche pour l'ancre astreinte, samedi pour l'ancre
        # garde) - forcé plus loin, section 9.
        if activity == "GARDE" and d_idx in (5, 6):
            is_combo_wom_member = req.weekend_astreinte_combo and doc_id in wom_pool
            garde_row = {"matin": "Garde Matin", "am": "Garde Midi", "nuit": "Garde Nuit"}.get(slot)
            is_manual_entry = (
                garde_row is not None
                and doc_id in (req.existing_schedule or {}).get(f"{garde_row}||{DAY_NAMES_FR[d_idx]}", [])
            )
            if not (is_combo_wom_member or is_manual_entry):
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
    combo_priority_bonus = []  # préférence souple pour les ancres combo weekend désignées
    full_day_hors_site_vars: List[tuple] = []  # (doc_id, d_idx, var) - appliqué après création de toutes les vars
    non_exclusive_activities: Set[str] = set()  # activités exemptées de "une activité par créneau" (ex: IRM)
    # Exemptions ciblées (médecin, jour, activité) - contrairement à
    # non_exclusive_activities (qui exempte TOUT LE MONDE sur cette
    # activité), ceci ne vaut que pour CE médecin précis ce jour précis (ex:
    # S peut cumuler ETT ped mercredi avec une garde, mais ça ne change rien
    # pour un autre médecin qui ferait "Apm - ETT salle 1" un autre jour via
    # le mécanisme générique). Confirmé utilisateur 29/07/2026.
    non_exclusive_doctor_day: Set[tuple] = set()
    irm_non_exclusive_pending: List[tuple] = []  # (doc_id, d_idx, slot, var) - exclusion Cs/ETT/Stress différée

    for row_key, config in HORS_SITE_CONFIG.items():
        allowed = config["allowed"]
        full_day = config["full_day"]
        priority = config.get("priority", [])
        fixed_slots_by_doctor = config.get("fixed_slots_by_doctor")
        preferred_slots_by_doctor = config.get("preferred_slots_by_doctor")
        activity_name = f"HORSSITE::{row_key}"

        if fixed_slots_by_doctor:
            # Créneaux fixes forcés par médecin (ex: Scinti R=mardi matin,
            # T=lundi+mercredi matin ; CDL V=mardi matin avec repli O). Un
            # seul médecin authentiquement présent par (jour, créneau) - si
            # plusieurs médecins ont le même (jour, créneau) fixe (ex: CDL
            # V et O tous les deux "mardi matin"), la priorité (config
            # "priority") désigne qui gagne réellement, l'autre reste à 0
            # ce jour-là (repli seulement si le prioritaire est absent).
            non_exclusive = config.get("non_exclusive", False)
            priority = config.get("priority", [])

            # Regrouper par (day_name, slot) -> liste de médecins concernés
            slot_candidates: Dict[tuple, List[str]] = {}
            for doc_id, slots in fixed_slots_by_doctor.items():
                for day_name, slot in slots:
                    slot_candidates.setdefault((day_name, slot), []).append(doc_id)

            for (day_name, slot), candidates in slot_candidates.items():
                d_idx = DAY_NAMES_FR.index(day_name)
                short_name = row_key.split(" - ", 1)[1] if " - " in row_key else row_key
                if is_activity_suspended(days[d_idx], short_name, req.activity_maintenance):
                    warnings.append(f"{row_key} suspendu(e) le {day_name} (maintenance) - non couvert cette semaine.")
                    continue
                # Ordonner selon la priorité si définie, sinon ordre donné
                ordered = sorted(
                    candidates,
                    key=lambda d: priority.index(d) if d in priority else len(priority)
                ) if priority else candidates

                winner = None
                for doc_id in ordered:
                    if doc_id not in medecins_map:
                        continue
                    if is_on_vacation(doc_id, days[d_idx], req.vacations) or is_on_vacation(doc_id, days[d_idx], req.congres):
                        continue
                    winner = doc_id
                    break

                if winner is None:
                    warnings.append(
                        f"{row_key} : aucun médecin disponible le {day_name} {slot} "
                        f"(congé de tous les candidats) - créneau fixe non couvert cette semaine."
                    )
                    continue

                for doc_id in candidates:
                    if doc_id not in medecins_map:
                        continue
                    var = model.NewBoolVar(f"horssite_{doc_id}_{d_idx}_{row_key}")
                    x[(doc_id, d_idx, slot, activity_name)] = var
                    model.Add(var == (1 if doc_id == winner else 0))
                    if full_day and doc_id == winner:
                        full_day_hors_site_vars.append((doc_id, d_idx, var))
                    if non_exclusive and doc_id == winner:
                        non_exclusive_activities.add(activity_name)
                        irm_non_exclusive_pending.append((doc_id, d_idx, slot, var))
            continue

        if preferred_slots_by_doctor:
            # Comme fixed_slots_by_doctor, mais OPTIONNEL (pas de model.Add
            # (var == 1)) : une préférence forte via bonus, pas une
            # obligation. Donne au solveur la liberté de laisser le créneau
            # non couvert si le(s) médecin(s) concerné(s) sont indisponibles,
            # plutôt qu'un avertissement "créneau fixe non couvert" (confirmé
            # utilisateur 29/07/2026).
            slot_candidates: Dict[tuple, List[str]] = {}
            for doc_id, slots in preferred_slots_by_doctor.items():
                for day_name, slot in slots:
                    slot_candidates.setdefault((day_name, slot), []).append(doc_id)

            for (day_name, slot), candidates in slot_candidates.items():
                d_idx = DAY_NAMES_FR.index(day_name)
                day_vars: Dict[str, Any] = {}
                for doc_id in candidates:
                    if doc_id not in medecins_map:
                        continue
                    if is_on_vacation(doc_id, days[d_idx], req.vacations) or is_on_vacation(doc_id, days[d_idx], req.congres):
                        continue
                    var = model.NewBoolVar(f"pref_{doc_id}_{d_idx}_{row_key}")
                    x[(doc_id, d_idx, slot, activity_name)] = var
                    day_vars[doc_id] = var
                    hors_site_priority_bonus.append(10 * var)  # préférence forte, pas absolue
                if day_vars:
                    model.Add(sum(day_vars.values()) <= 1)
                    hors_site_vars[(row_key, d_idx)] = day_vars
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

    # --- LFB / PSSL : rotations désignées en entrée (voir req.lfb_doctor /
    # pssl_b_active / pssl_z_active) - PAS calculées par le solveur (même
    # raisonnement que VISITE). Toutes deux "journée entière" (bloquent aussi
    # l'après-midi, cf. full_day_hors_site_vars).
    if req.lfb_doctor:
        jeudi_idx = DAY_NAMES_FR.index("JEUDI")
        LFB_POOL = ["H", "S", "G"]
        if is_activity_suspended(days[jeudi_idx], "LFB", req.activity_maintenance):
            warnings.append("LFB suspendu ce jeudi (maintenance) - non couvert cette semaine.")
        elif req.lfb_doctor not in LFB_POOL:
            warnings.append(f"lfb_doctor '{req.lfb_doctor}' invalide (attendu H, S ou G) - ignoré.")
        else:
            # Repli automatique sur les 2 autres du pool si le désigné est
            # indisponible (confirmé utilisateur 29/07/2026, donne plus de
            # marge au solveur plutôt que de laisser le créneau non couvert).
            candidates = [req.lfb_doctor] + [d for d in LFB_POOL if d != req.lfb_doctor]
            winner = None
            for doc in candidates:
                if doc not in medecins_map:
                    continue
                if is_on_vacation(doc, days[jeudi_idx], req.vacations) or is_on_vacation(doc, days[jeudi_idx], req.congres):
                    continue
                winner = doc
                break
            if winner is None:
                warnings.append(f"LFB : aucun de H/S/G disponible ce jeudi - créneau non couvert cette semaine.")
            else:
                var = model.NewBoolVar(f"lfb_{winner}_{jeudi_idx}")
                x[(winner, jeudi_idx, "matin", "HORSSITE::Hors site - LFB")] = var
                model.Add(var == 1)
                full_day_hors_site_vars.append((winner, jeudi_idx, var))
                if winner != req.lfb_doctor:
                    warnings.append(f"LFB : {req.lfb_doctor} indisponible ce jeudi, repli sur {winner}.")

    if req.pssl_doctor:
        jeudi_idx = DAY_NAMES_FR.index("JEUDI")
        PSSL_POOL = ["B", "Z"]
        if is_activity_suspended(days[jeudi_idx], "PSSL", req.activity_maintenance):
            warnings.append("PSSL suspendu ce jeudi (maintenance) - non couvert cette semaine.")
        elif req.pssl_doctor not in PSSL_POOL:
            warnings.append(f"pssl_doctor '{req.pssl_doctor}' invalide (attendu B ou Z) - ignoré.")
        else:
            candidates = [req.pssl_doctor] + [d for d in PSSL_POOL if d != req.pssl_doctor]
            winner = None
            for doc in candidates:
                if doc not in medecins_map:
                    continue
                if is_on_vacation(doc, days[jeudi_idx], req.vacations) or is_on_vacation(doc, days[jeudi_idx], req.congres):
                    continue
                winner = doc
                break
            if winner is None:
                warnings.append("PSSL : ni B ni Z disponible ce jeudi - créneau non couvert cette semaine.")
            else:
                var = model.NewBoolVar(f"pssl_{winner}_{jeudi_idx}")
                x[(winner, jeudi_idx, "matin", "HORSSITE::Hors site - PSSL")] = var
                model.Add(var == 1)
                full_day_hors_site_vars.append((winner, jeudi_idx, var))
                if winner != req.pssl_doctor:
                    warnings.append(f"PSSL : {req.pssl_doctor} indisponible ce jeudi, repli sur {winner}.")

    # --- Coro mercredi après-midi : O par défaut, repli libre M/W si absent
    # (confirmé utilisateur 31/07/2026 : "O fait tout le temps la coro
    # mercredi AM sauf vacances", pas de priorité entre M/W au repli - le
    # solveur choisit librement). Créé directement (comme FIXED_CS_ETT_SLOTS)
    # pour être robuste même sans historical_patterns.
    mercredi_idx = DAY_NAMES_FR.index("MERCREDI")
    mercredi_am_maintenance = is_room_under_maintenance(days[mercredi_idx], "am", req.room_maintenance)
    if mercredi_am_maintenance:
        warnings.append("Coro mercredi après-midi : salle en maintenance - créneau non couvert.")
    elif "O" in medecins_map and not (
        is_on_vacation("O", days[mercredi_idx], req.vacations) or is_on_vacation("O", days[mercredi_idx], req.congres)
    ):
        var_o = x.get(("O", mercredi_idx, "am", "CORO"))
        if var_o is None:
            var_o = model.NewBoolVar(f"coro_mercredi_o_{mercredi_idx}")
            x[("O", mercredi_idx, "am", "CORO")] = var_o
        model.Add(var_o == 1)
        for other in ("M", "W"):
            v_other = x.get((other, mercredi_idx, "am", "CORO"))
            if v_other is not None:
                model.Add(v_other == 0)
    else:
        warnings.append("Coro mercredi après-midi : O en congé - repli libre entre M et W.")
        for other in ("M", "W"):
            if other not in medecins_map:
                continue
            if is_on_vacation(other, days[mercredi_idx], req.vacations) or is_on_vacation(other, days[mercredi_idx], req.congres):
                continue
            v_other = x.get((other, mercredi_idx, "am", "CORO"))
            if v_other is None:
                # M/W sont normalement en demi-journée fixe libre mercredi
                # après-midi - exception explicite pour ce repli (confirmé
                # utilisateur 31/07/2026 : l'un d'eux doit pouvoir couvrir
                # Coro quand O est absent, malgré leur 1/2 off habituelle).
                v_other = model.NewBoolVar(f"coro_mercredi_repli_{other}")
                x[(other, mercredi_idx, "am", "CORO")] = v_other
            hors_site_priority_bonus.append(50 * v_other)

    # --- Coro vendredi (matin + après-midi) : alternance M/W selon parité de
    # semaine (confirmé utilisateur 31/07/2026, O jamais candidat ce jour) :
    # semaine impaire -> W matin ET après-midi ; semaine paire -> M matin, W
    # après-midi. Repli sur l'autre du couple M/W si le titulaire du jour est
    # indisponible (même principe que LFB/PSSL).
    vendredi_idx = DAY_NAMES_FR.index("VENDREDI")
    if req.week_type == 1:
        vendredi_coro_slots = [("matin", "W"), ("am", "M")]
    else:
        vendredi_coro_slots = [("matin", "M"), ("am", "W")]
    for slot, preferred in vendredi_coro_slots:
        if is_room_under_maintenance(days[vendredi_idx], slot, req.room_maintenance):
            warnings.append(f"Coro vendredi {slot} : salle en maintenance - créneau non couvert.")
            continue
        candidates = [preferred] + [d for d in ("M", "W") if d != preferred]
        winner = None
        for doc in candidates:
            if doc not in medecins_map:
                continue
            if is_on_vacation(doc, days[vendredi_idx], req.vacations) or is_on_vacation(doc, days[vendredi_idx], req.congres):
                continue
            winner = doc
            break
        if winner is None:
            warnings.append(f"Coro vendredi {slot} : ni M ni W disponible - créneau non couvert cette semaine.")
            continue
        var_winner = x.get((winner, vendredi_idx, slot, "CORO"))
        if var_winner is None:
            var_winner = model.NewBoolVar(f"coro_vendredi_{winner}_{slot}")
            x[(winner, vendredi_idx, slot, "CORO")] = var_winner
        model.Add(var_winner == 1)
        for other in ("M", "W"):
            if other == winner:
                continue
            v_other = x.get((other, vendredi_idx, slot, "CORO"))
            if v_other is not None:
                model.Add(v_other == 0)
        # O jamais candidat sur Coro vendredi (confirmé utilisateur)
        v_o = x.get(("O", vendredi_idx, slot, "CORO"))
        if v_o is not None:
            model.Add(v_o == 0)

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
                # Restrictions strictes de périmètres par activité (Cs PSS, Cs Tessée, ETT, EE, ETT Tessée)
                if row_key.endswith("Cs PSS") and doc_id not in CS_PSS_ALLOWED:
                    continue
                if row_key.endswith("Cs Tessée") and doc_id not in CS_TESSEE_ALLOWED:
                    continue
                if ("ETT salle" in row_key or row_key.startswith("Matin - ETT") or row_key.startswith("Apm - ETT")) and doc_id not in ETT_ALLOWED:
                    continue
                if ("EE" in row_key) and doc_id not in EE_ALLOWED:
                    continue
                if ("ETT Tessée" in row_key) and doc_id not in ETT_TESSEE_ALLOWED:
                    continue
                # VISITE (A/B/U en roulement, désigné en entrée) : pas de Cs le
                # matin cette semaine pour le médecin en visite (confirmé
                # utilisateur 28/07/2026). Ne s'applique qu'au matin, pas
                # après-midi, et qu'aux lignes "Cs" (pas ETT/Stress/EE).
                if (req.visite_doctor and doc_id == req.visite_doctor
                        and slot == "matin" and row_key.startswith("Matin - Cs")):
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

    # --- Créneaux fixes Cs PM / ETT ped (confirmé utilisateur 28/07/2026) ---
    # S : ETT ped (écho pédiatrique) mercredi après-midi, sur "Apm - ETT salle 1".
    # A : Cs PM (contrôle PM) mardi après-midi, sur "Apm - Cs PSS" (A=PSS uniquement).
    # P : Cs PM (contrôle PM) lundi après-midi, sur "Apm - Cs PSS" (P=PSS uniquement).
    # Robuste même si historical_patterns ne mentionne pas ces créneaux (créé
    # directement ici, pas dépendant du mécanisme de fréquence historique) -
    # à la différence de U (Cs PM sans jour fixe), qui reste géré par le
    # mécanisme générique existant.
    FIXED_CS_ETT_SLOTS = [
        ("S", "MERCREDI", "am", "Apm - ETT salle 1"),
        ("A", "MARDI", "am", "Apm - Cs PSS"),
        ("P", "LUNDI", "am", "Apm - Cs PSS"),
    ]
    # S peut cumuler ETT ped avec une autre tâche (garde, etc.) sur le même
    # créneau - confirmé utilisateur 29/07/2026. PAS étendu à A/P pour
    # l'instant (confirmé explicitement : seulement S).
    NON_EXCLUSIVE_FIXED_DOCTORS = {"S"}
    for doc_id, day_name, slot, row_key in FIXED_CS_ETT_SLOTS:
        if doc_id not in medecins_map:
            continue
        d_idx = DAY_NAMES_FR.index(day_name)
        if is_on_vacation(doc_id, days[d_idx], req.vacations) or is_on_vacation(doc_id, days[d_idx], req.congres):
            warnings.append(f"{row_key} ({day_name}) : {doc_id} en congé - créneau fixe non couvert cette semaine.")
            continue

        activity_name = f"HIST::{row_key}"
        var = x.get((doc_id, d_idx, slot, activity_name))
        if var is None:
            var = model.NewBoolVar(f"fixed_{doc_id}_{d_idx}_{row_key}")
            x[(doc_id, d_idx, slot, activity_name)] = var
        model.Add(var == 1)

        # Réserve exclusivement ce créneau à ce médecin : force à 0 tout
        # autre médecin qui aurait aussi une variable sur cette même case
        # (ex: si historical_patterns proposait quelqu'un d'autre ici).
        for (d2, dd2, sl2, act2), v2 in list(x.items()):
            if dd2 == d_idx and sl2 == slot and act2 == activity_name and d2 != doc_id:
                model.Add(v2 == 0)

        if doc_id in NON_EXCLUSIVE_FIXED_DOCTORS:
            # Cumul possible avec garde/astreinte sur ce même créneau
            # (confirmé utilisateur 29/07/2026) - même mécanisme que IRM :
            # exempté de "une activité par créneau", SAUF vis-à-vis d'un
            # autre Cs/ETT/Stress (physiquement impossible en même temps).
            non_exclusive_activities.add(activity_name)
            irm_non_exclusive_pending.append((doc_id, d_idx, slot, var))

    # --- Binôme infirmière/médecin sur Stress/EE (confirmé utilisateur
    # 31/07/2026) : le front positionne l'infirmière (Val/Véro/Laura) via
    # existing_schedule ; le solveur propose ici le médecin partenaire du
    # pool correspondant, pour chaque case où une infirmière est déjà
    # présente. Bonus fort (pas une obligation dure) pour rester robuste si
    # personne du pool n'est disponible ce jour-là.
    for row_key in STRESS_ROWS | EE_ROWS:
        slot = "matin" if row_key.startswith("Matin") else "am"
        pool = STRESS_PARTNER_POOL if row_key in STRESS_ROWS else EE_PARTNER_POOL
        for day_name in DAY_NAMES_FR[:5]:
            existing_here = (req.existing_schedule or {}).get(f"{row_key}||{day_name}", [])
            if not any(n in existing_here for n in NURSES):
                continue  # pas d'infirmière ici, rien à proposer
            if any(d for d in existing_here if d not in NURSES):
                continue  # un partenaire est déjà saisi manuellement, ne pas y toucher
            d_idx = DAY_NAMES_FR.index(day_name)
            activity_name = f"HIST::{row_key}"
            partner_vars = []
            for doc_id in pool:
                if doc_id not in medecins_map:
                    continue
                if is_on_vacation(doc_id, days[d_idx], req.vacations) or is_on_vacation(doc_id, days[d_idx], req.congres):
                    continue
                var = x.get((doc_id, d_idx, slot, activity_name))
                if var is None:
                    var = model.NewBoolVar(f"nurse_partner_{doc_id}_{d_idx}_{row_key}")
                    x[(doc_id, d_idx, slot, activity_name)] = var
                partner_vars.append(var)
            if partner_vars:
                model.Add(sum(partner_vars) <= 1)
                has_partner = model.NewBoolVar(f"has_partner_{d_idx}_{row_key}")
                model.Add(sum(partner_vars) >= 1).OnlyEnforceIf(has_partner)
                model.Add(sum(partner_vars) == 0).OnlyEnforceIf(has_partner.Not())
                hors_site_priority_bonus.append(40 * has_partner)
            else:
                warnings.append(
                    f"{row_key} ({day_name}) : aucun médecin du pool partenaire disponible "
                    f"pour accompagner l'infirmière - créneau incomplet cette semaine."
                )

    # --- Doublon Cs (préférence forte, pas absolue - confirmé utilisateur
    # 29/07/2026) : Z lundi après-midi, H mardi après-midi - le même médecin
    # sur 2 salles/sessions simultanées, affiché "Z²"/"H²" côté front (déjà
    # géré par lib/slot-blocking.ts). N'entre PAS en conflit avec la capacité
    # normale "<=1" de la case (doublon_var est une couche additionnelle, pas
    # comptée dans cette somme) - le médecin doit déjà être présent une fois
    # (base_var) pour pouvoir être en doublon.
    DOUBLON_CS_CONFIG = [
        ("Z", "LUNDI", "am", "Apm - Cs PSS"),
        ("H", "MARDI", "am", "Apm - Cs PSS"),
    ]
    doublon_bonus_terms = []
    doublon_output_pairs: List[tuple] = []  # (doc_id, d_idx, slot, row_key, doublon_var)
    for doc_id, day_name, slot, row_key in DOUBLON_CS_CONFIG:
        if doc_id not in medecins_map:
            continue
        d_idx = DAY_NAMES_FR.index(day_name)
        if is_on_vacation(doc_id, days[d_idx], req.vacations) or is_on_vacation(doc_id, days[d_idx], req.congres):
            continue
        activity_name = f"HIST::{row_key}"
        base_var = x.get((doc_id, d_idx, slot, activity_name))
        if base_var is None:
            base_var = model.NewBoolVar(f"cspm_base_{doc_id}_{d_idx}_{row_key}")
            x[(doc_id, d_idx, slot, activity_name)] = base_var
        doublon_var = model.NewBoolVar(f"doublon_{doc_id}_{d_idx}_{row_key}")
        model.Add(doublon_var <= base_var)  # doublon seulement si déjà présent une fois
        doublon_bonus_terms.append(5 * doublon_var)
        doublon_output_pairs.append((doc_id, d_idx, slot, row_key, doublon_var))
    doublon_bonus = sum(doublon_bonus_terms) if doublon_bonus_terms else 0

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

    # --- Application différée : IRM peut cumuler une autre activité au même
    # créneau (garde, astreinte...), SAUF Cs/ETT/Stress, explicitement exclus
    # (confirmé utilisateur 28/07/2026). Fait ici, une fois les variables
    # Cs/ETT/Stress (section 2bis) créées.
    for doc_id, d_idx, slot, irm_var in irm_non_exclusive_pending:
        cs_ett_stress_vars_same_slot = [
            v for (d2, dd2, sl2, act2), v in x.items()
            if d2 == doc_id and dd2 == d_idx and sl2 == slot
            and act2.startswith("HIST::")
            and re.match(r"^(Matin|Apm) - (Cs |ETT |Stress$)", act2[len("HIST::"):])
            and v is not irm_var  # évite l'auto-exclusion si le créneau non-exclusif est lui-même Cs/ETT/Stress (ex: ETT ped de S)
        ]
        for cev in cs_ett_stress_vars_same_slot:
            model.Add(cev == 0).OnlyEnforceIf(irm_var)

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
    # Stress/Garde Midi ce même créneau - voir 2quater. IRM également exceptée
    # depuis le 28/07/2026 : S peut cumuler garde/astreinte en plus d'IRM sur
    # le même créneau - seule l'exclusion Cs/ETT/Stress avec IRM reste bloquée,
    # gérée séparément ci-dessus. Aucune de ces deux n'est une "vraie"
    # activité concurrente pour le temps du médecin.)
    #
    # ATL Matin/Midi Lun–Ven == Coro (section 10) : même présence physique,
    # deux BoolVar forcés égaux. Les compter tous les deux dans l'exclusivité
    # rend CORO+ASTREINTE=1 impossible → INFEASIBLE dès que FV (ou tout
    # coronarographiste) a les deux vars (ex. rules_override astreinte_allowed
    # avec FV, confirmé 29/07/2026 : "Aucune solution trouvée" + LFB en repli).
    # On n'exclut ASTREINTE que si une var CORO existe sur le même créneau.
    for doc_id in medecins_map:
        for d_idx in range(7):
            for slot in SLOTS:
                slot_vars = [
                    v for (doc, d, sl, act), v in x.items()
                    if d == d_idx and sl == slot and doc == doc_id
                    and act != entrees_pss_activity and act not in non_exclusive_activities
                    and not (
                        act == "ASTREINTE"
                        and slot in ("matin", "am")
                        and d_idx < 5
                        and x.get((doc_id, d_idx, slot, "CORO")) is not None
                    )
                ]
                if slot_vars:
                    model.Add(sum(slot_vars) <= 1)

    # --- 3bis. Règle dynamique : Repos et enchaînements après Garde / Astreinte de Nuit ---
    # Règle GARDE NUIT (stricte) :
    # Un médecin qui fait une GARDE de nuit (act == "GARDE") le jour d ne peut PAS
    # faire Coro, Astreinte ATL ou Rythmo le lendemain (d+1).
    # Si ce médecin est le seul éligible/disponible pour la Coro ou Rythmo le lendemain,
    # CP-SAT choisira UN AUTRE MÉDECIN pour la GARDE de nuit le jour d.
    #
    # Règle ASTREINTE NUIT (souple) :
    # Ne bloque JAMAIS la Coro ni l'Astreinte ATL le lendemain après-midi.
    # Évite seulement de préférence (pénalité souple) la Coro / Astreinte ATL le matin du lendemain.
    post_night_guard_off_flags: Dict[tuple, Any] = {}  # (doc_id, d_idx) -> BoolVar "a fait une garde de nuit ce jour-là"
    astreinte_nuit_coro_matin_penalties = []

    for doc_id in medecins_map:
        for d_idx in range(6):  # LUNDI(0) à SAMEDI(5)
            # 1) GARDE NUIT (Stricte)
            night_garde_vars = [
                v for (doc, d, sl, act), v in x.items()
                if doc == doc_id and d == d_idx and sl == "nuit" and act == "GARDE"
            ]
            if night_garde_vars:
                worked_night_garde = model.NewBoolVar(f"worked_night_garde_{doc_id}_{d_idx}")
                model.Add(sum(night_garde_vars) >= 1).OnlyEnforceIf(worked_night_garde)
                model.Add(sum(night_garde_vars) == 0).OnlyEnforceIf(worked_night_garde.Not())
                post_night_guard_off_flags[(doc_id, d_idx)] = worked_night_garde

                if d_idx < 4:
                    next_day_name = DAY_NAMES_FR[d_idx + 1]
                    target_slot = target_off_slot_after_night_guard(doc_id, next_day_name)
                    other_vars_next_day = [
                        v for (doc, d, sl, act), v in x.items()
                        if doc == doc_id and d == d_idx + 1 and sl == target_slot
                    ]
                    for v in other_vars_next_day:
                        model.Add(v == 0).OnlyEnforceIf(worked_night_garde)

                # Interdiction Coro / Astreinte ATL / Rythmo le lendemain d'une GARDE de nuit.
                # RÈGLE SPÉCIALE M, O, W : Ne s'applique QUE SI les 3 (M, O, W) sont présents aujourd'hui (d) et demain (d+1).
                # Si 1 ou 2 d'entre eux est absent/en congés, la règle tombe automatiquement et l'enchaînement est autorisé.
                mow_all_present_today_and_next = (
                    d_idx < 6 and all(
                        not is_on_vacation(m_code, days[d_idx], req.vacations) and
                        not is_on_vacation(m_code, days[d_idx + 1], req.vacations) and
                        not (m_code in fixed_exclusions and d_idx in fixed_exclusions[m_code]) and
                        not (m_code in fixed_exclusions and (d_idx + 1) in fixed_exclusions[m_code])
                        for m_code in ("M", "O", "W")
                    )
                )

                coro_rythmo_next_day_vars = []
                for (doc, d, sl, act), v in x.items():
                    if doc == doc_id and d == d_idx + 1:
                        is_coro_atl = (
                            act in ("CORO", "ASTREINTE") or
                            act.startswith("HIST::Matin - Coro") or
                            act.startswith("HIST::Apm - Coro")
                        )
                        is_rythmo = (
                            act == "RYTHMO" or
                            act.startswith("HIST::Matin - Rythmo") or
                            act.startswith("HIST::Apm - Rythmo")
                        )

                        if is_rythmo:
                            coro_rythmo_next_day_vars.append(v)
                        elif is_coro_atl:
                            if doc_id in ("M", "O", "W"):
                                if mow_all_present_today_and_next:
                                    coro_rythmo_next_day_vars.append(v)
                            else:
                                coro_rythmo_next_day_vars.append(v)

                for v in coro_rythmo_next_day_vars:
                    model.Add(v == 0).OnlyEnforceIf(worked_night_garde)

            # 2) ASTREINTE NUIT (Souple le matin, autorisée l'après-midi)
            night_astreinte_vars = [
                v for (doc, d, sl, act), v in x.items()
                if doc == doc_id and d == d_idx and sl == "nuit" and act == "ASTREINTE"
            ]
            if night_astreinte_vars:
                worked_night_ast = model.NewBoolVar(f"worked_night_ast_{doc_id}_{d_idx}")
                model.Add(sum(night_astreinte_vars) >= 1).OnlyEnforceIf(worked_night_ast)
                model.Add(sum(night_astreinte_vars) == 0).OnlyEnforceIf(worked_night_ast.Not())

                coro_matin_next_day = [
                    v for (doc, d, sl, act), v in x.items()
                    if doc == doc_id and d == d_idx + 1 and sl == "matin" and (
                        act in ("CORO", "ASTREINTE") or
                        act.startswith("HIST::Matin - Coro")
                    )
                ]
                for v in coro_matin_next_day:
                    penalty_var = model.NewBoolVar(f"pen_ast_coro_{doc_id}_{d_idx}")
                    model.AddBoolAnd([worked_night_ast, v]).OnlyEnforceIf(penalty_var)
                    model.AddBoolOr([worked_night_ast.Not(), v.Not()]).OnlyEnforceIf(penalty_var.Not())
                    astreinte_nuit_coro_matin_penalties.append(10 * penalty_var)
            # Interdiction stricte Coro / Astreinte ATL / Rythmo le lendemain d'une GARDE de nuit.
            # Si le médecin est le seul disponible/éligible pour la Coro ou le Rythmo le lendemain,
            # cette contrainte interdira à ce médecin de faire la garde de nuit la veille,
            # forçant le solveur à choisir UN AUTRE MÉDECIN pour la garde de nuit.
            if night_garde_vars:
                coro_rythmo_next_day_vars = [
                    v for (doc, d, sl, act), v in x.items()
                    if doc == doc_id and d == d_idx + 1 and (
                        act in ("CORO", "ASTREINTE", "RYTHMO") or
                        act.startswith("HIST::Matin - Coro") or
                        act.startswith("HIST::Apm - Coro") or
                        act.startswith("HIST::Matin - Rythmo") or
                        act.startswith("HIST::Apm - Rythmo")
                    )
                ]
                for v in coro_rythmo_next_day_vars:
                    model.Add(v == 0).OnlyEnforceIf(worked_night_garde)

    # Cas dimanche (semaine précédente) -> lundi (cette semaine) : le doctor est connu
    # à l'avance (transmis par le front), donc traité comme une exclusion fixe
    # classique plutôt qu'une réification - pas besoin de deviner, on SAIT déjà que
    # ce médecin a fait la garde/astreinte dimanche dernier.
    if req.previous_sunday_guard_doctor:
        sunday_doc = req.previous_sunday_guard_doctor
        monday_name = DAY_NAMES_FR[0]
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

    # Astreintes ATL Midi fermées de S31 à S34 inclus de Lundi à Vendredi
    week_start_d = date.fromisoformat(req.week_start_date)
    week_num = week_start_d.isocalendar()[1]
    if 31 <= week_num <= 34:
        for d_idx in range(5):
            for doc in medecins_map:
                var = x.get((doc, d_idx, "am", "ASTREINTE"))
                if var is not None:
                    model.Add(var == 0)

    # --- 4ter. Couverture quotidienne obligatoire Coro matin/am (lundi-
    # vendredi) - confirmé utilisateur 31/07/2026 : contrairement à la nuit
    # (structurée ci-dessus), le jour n'avait AUCUNE obligation de couverture
    # - seulement une incitation d'équité, qui pouvait laisser un créneau
    # totalement vide même quand M/O/W étaient tous disponibles (bug
    # constaté S35). Compatible avec les forçages déjà en place plus loin
    # (mercredi après-midi, vendredi) qui satisfont déjà "exactement 1".
    for d_idx in range(5):
        for slot in ("matin", "am"):
            coro_allowed_vars = [
                var for (doc, d, sl, act), var in x.items()
                if d == d_idx and sl == slot and act == "CORO" and doc in CORO_ALLOWED
            ]
            if coro_allowed_vars:
                has_coverage = model.NewBoolVar(f"coro_daily_coverage_{d_idx}_{slot}")
                model.Add(sum(coro_allowed_vars) >= 1).OnlyEnforceIf(has_coverage)
                model.Add(sum(coro_allowed_vars) == 0).OnlyEnforceIf(has_coverage.Not())
                model.Add(sum(coro_allowed_vars) <= 1)
                hors_site_priority_bonus.append(40 * has_coverage)
            else:
                warnings.append(
                    f"Coro {DAY_NAMES_FR[d_idx]} {slot} : aucun médecin disponible - "
                    f"créneau non couvert cette semaine."
                )

    # --- 4quater. Coro matin ≠ Coro après-midi le même jour, UNIQUEMENT si
    # les 3 coronarographistes (M, O, W) sont disponibles ce jour-là -
    # confirmé utilisateur 31/07/2026 : "de préférence 2 médecins par jour
    # (1 par vacation)... seulement si absence d'un ou des 2 autres" (1 seul
    # médecin peut alors couvrir les deux créneaux). Lundi-vendredi
    # uniquement (le weekend n'est pas concerné par cette règle).
    for d_idx in range(5):
        wom_present_today = [
            doc for doc in wom_pool
            if not is_on_vacation(doc, days[d_idx], req.vacations)
            and not is_on_vacation(doc, days[d_idx], req.congres)
        ]
        if len(wom_present_today) < 3:
            # Absence(s) : pas de restriction dure, MAIS préférence pour
            # qu'un seul médecin couvre les deux créneaux ce jour-là
            # (confirmé utilisateur 31/07/2026), plutôt que 2 personnes
            # différentes par défaut faute d'incitation contraire.
            for doc in wom_present_today:
                matin_var = x.get((doc, d_idx, "matin", "CORO"))
                am_var = x.get((doc, d_idx, "am", "CORO"))
                if matin_var is not None and am_var is not None:
                    same_doc_both_slots = model.NewBoolVar(f"coro_same_doc_{doc}_{d_idx}")
                    model.AddBoolAnd([matin_var, am_var]).OnlyEnforceIf(same_doc_both_slots)
                    model.AddBoolOr([matin_var.Not(), am_var.Not()]).OnlyEnforceIf(same_doc_both_slots.Not())
                    hors_site_priority_bonus.append(25 * same_doc_both_slots)
            continue
        for doc in wom_pool:
            matin_var = x.get((doc, d_idx, "matin", "CORO"))
            am_var = x.get((doc, d_idx, "am", "CORO"))
            if matin_var is not None and am_var is not None:
                model.Add(matin_var + am_var <= 1)

    # --- 4bis. M/O/W ne peuvent jamais faire 2 astreintes de nuit la même
    # semaine en semaine (lundi-vendredi) - confirmé utilisateur 27/07/2026 :
    # "jamais 2 fois le même médecin", total sur la semaine (pas seulement
    # consécutif). Le weekend (samedi/dimanche) est explicitement exempté de
    # cette règle - un même médecin WOM peut y être présent sans que ça compte.
    #
    # Assouplissement automatique (confirmé utilisateur 31/07/2026) : si un
    # seul membre du pool WOM est disponible cette semaine (les 2 autres
    # totalement absents), la règle est impossible à respecter (il faut
    # couvrir plusieurs nuits WOM avec une seule personne) - on la relâche
    # UNIQUEMENT pour ce médecin-là, avec un avertissement explicite.
    _wom_weekday_vars_by_doc = {
        doc: [
            v for (d_op, d, sl, act), v in x.items()
            if d_op == doc and d < 5 and sl == "nuit" and act == "ASTREINTE"
        ]
        for doc in wom_pool
    }
    _wom_available_this_week = [doc for doc, vs in _wom_weekday_vars_by_doc.items() if vs]
    for doc in wom_pool:
        weekday_night_vars = _wom_weekday_vars_by_doc[doc]
        if not weekday_night_vars:
            continue
        if len(_wom_available_this_week) == 1:
            warnings.append(
                f"⚠️ Seul {doc} est disponible cette semaine parmi les coronarographistes "
                f"(les autres sont absents) - règle \"jamais 2 astreintes de nuit/semaine\" "
                f"assouplie exceptionnellement pour lui, plusieurs nuits possibles."
            )
            continue
        model.Add(sum(weekday_night_vars) <= 1)

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

        # Alternance NCT : ne pas répéter le même que la semaine précédente -
        # sauf si un seul candidat NCT reste disponible cette semaine (l'autre
        # étant absent), auquel cas la répétition est inévitable (confirmé
        # utilisateur 31/07/2026, même principe que l'assouplissement
        # astreinte hebdomadaire ci-dessus).
        _nct_available_this_week = [
            doc for doc in nct_pool if x.get((doc, 3, "nuit", "NCT")) is not None
        ]
        if req.last_nct_doctor and req.last_nct_doctor in nct_pool:
            if len(_nct_available_this_week) == 1 and _nct_available_this_week[0] == req.last_nct_doctor:
                warnings.append(
                    f"⚠️ Seul {req.last_nct_doctor} est disponible pour la NCT cette semaine "
                    f"(l'autre est absent) - alternance non respectée exceptionnellement, "
                    f"répétition inévitable."
                )
            else:
                var_nct = x.get((req.last_nct_doctor, 3, "nuit", "NCT"))
                if var_nct is not None:
                    model.Add(var_nct == 0)

        # NCT interdit si astreinte nuit ou garde nuit la veille (mercredi)
        for doc in nct_pool:
            if len(_nct_available_this_week) == 1 and _nct_available_this_week[0] == doc:
                continue
            var_nct = x.get((doc, 3, "nuit", "NCT"))
            var_astreinte_mercredi = x.get((doc, 2, "nuit", "ASTREINTE"))
            if var_nct is not None and var_astreinte_mercredi is not None:
                model.AddImplication(var_nct, var_astreinte_mercredi.Not())
            var_garde_mercredi = x.get((doc, 2, "nuit", "GARDE"))
            if var_nct is not None and var_garde_mercredi is not None:
                model.AddImplication(var_nct, var_garde_mercredi.Not())

    # --- 5bis. REEDUC (obligatoire, 1 médecin exactement, Lundi/Mercredi/Vendredi am) ---
    # Mercredi : S fortement privilégié, R/K seulement en repli si S
    # indisponible (confirmé utilisateur 28/07/2026) - même mécanisme que la
    # priorité CDL (V>O) : bonus fort pour S, pas une exclusion de R/K, pour
    # que le solveur les choisisse quand même si S est vraiment absent.
    reeduc_priority_bonus = []
    for d_idx, day_nm in enumerate(DAY_NAMES_FR):
        if day_nm not in REEDUC_DAYS:
            continue
        reeduc_vars = [v for (doc, d, sl, act), v in x.items() if d == d_idx and sl == "am" and act == "REEDUC"]
        if reeduc_vars:
            model.Add(sum(reeduc_vars) == 1)
            if day_nm == "MERCREDI":
                for (doc, d, sl, act), v in x.items():
                    if d == d_idx and sl == "am" and act == "REEDUC":
                        weight = 100 if doc == "S" else 1
                        reeduc_priority_bonus.append(weight * v)
        else:
            warnings.append(f"{day_nm} : aucun médecin disponible pour REEDUC (vacances ou exclu)")

    # --- 6. Fixes forcés (FV) ---
    if fv_id:
        fv_thu_vac = is_on_vacation("FV", days[3], req.vacations) or is_on_vacation("FV", days[3], req.congres)
        if not fv_thu_vac:
            var_fv_coro = x.get(("FV", 3, "am", "CORO"))
            if var_fv_coro is not None:
                model.Add(var_fv_coro == 1)

        fv_mon_vac = is_on_vacation("FV", days[0], req.vacations) or is_on_vacation("FV", days[0], req.congres)
        if not fv_mon_vac and req.fv_monday_night_active:
            var_fv_garde = x.get(("FV", 0, "nuit", "GARDE"))
            if var_fv_garde is not None:
                model.Add(var_fv_garde == 1)
        elif fv_mon_vac and req.fv_monday_night_active:
            # Repli sur U si FV est absent le lundi soir
            var_u_garde = x.get(("U", 0, "nuit", "GARDE"))
            if var_u_garde is not None and not (is_on_vacation("U", days[0], req.vacations) or is_on_vacation("U", days[0], req.congres)):
                model.Add(var_u_garde == 1)

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

    for doc_id, day_name, slot in RYTHMO_FORCE:
        _force_rythmo(doc_id, day_name, slot)

    # --- 7. Règles d'exclusion métier ---
    # 7.1 retirée (28/07/2026) : redondante avec la section 3bis (repos
    # dynamique après garde de nuit) et incorrecte - elle bloquait
    # inconditionnellement le MATIN du lendemain, alors que la vraie règle
    # métier confirmée libère l'APRÈS-MIDI (voir target_off_slot_after_night_
    # guard). Cette section entrait en conflit avec toute activité fixe du
    # matin (ex: LFB en repli) le lendemain d'une garde de nuit, causant des
    # "Aucune solution trouvée" évitables.

    # 7.2 Garde nuit => pas d'ACTIVITÉ AUTRE sur AM le même jour (exclut GARDE
    # elle-même : un médecin peut légitimement faire GARDE nuit+am+matin le
    # même jour d'affilée - ex: weekend combo M/O/W - sans que ce soit
    # "une autre activité". Bug corrigé le 30/07/2026, même classe que le
    # correctif précédent sur l'exclusion Cs/ETT/Stress.)
    for doc_id in medecins_map:
        for d_idx in range(7):
            var_nuit_garde = x.get((doc_id, d_idx, "nuit", "GARDE"))
            if var_nuit_garde is None:
                continue
            am_vars = [
                v for (doc, d, sl, act), v in x.items()
                if d == d_idx and sl == "am" and doc == doc_id and act != "GARDE"
            ]
            if am_vars:
                presence_am = model.NewBoolVar(f"presence_am_{doc_id}_{d_idx}")
                model.Add(sum(am_vars) >= 1).OnlyEnforceIf(presence_am)
                model.Add(sum(am_vars) == 0).OnlyEnforceIf(presence_am.Not())
                model.AddImplication(var_nuit_garde, presence_am.Not())

    # 7.3 Pas d'astreinte nuit si garde ce jour (LUNDI-VENDREDI uniquement -
    # le weekend est couvert par la règle 9bis, plus bas, qui respecte les
    # saisies manuelles existantes ; cette règle-ci n'avait pas cette
    # exception et entrait en conflit avec elles, confirmé bug 30/07/2026).
    for doc_id in medecins_map:
        for d_idx in range(5):
            garde_vars = [v for (doc, d, sl, act), v in x.items() if d == d_idx and doc == doc_id and act == "GARDE"]
            if not garde_vars:
                continue
            garde_present = model.NewBoolVar(f"garde_present_{doc_id}_{d_idx}")
            model.Add(sum(garde_vars) >= 1).OnlyEnforceIf(garde_present)
            model.Add(sum(garde_vars) == 0).OnlyEnforceIf(garde_present.Not())
            nuit_astreinte = x.get((doc_id, d_idx, "nuit", "ASTREINTE"))
            if nuit_astreinte is not None:
                model.AddImplication(garde_present, nuit_astreinte.Not())

    # --- 8. Séquences valides pour M, O, W (LUNDI-VENDREDI uniquement - le
    # weekend est couvert par la règle 9bis, plus bas, qui respecte les
    # saisies manuelles existantes ; cette contrainte-ci n'avait pas cette
    # exception et entrait en conflit avec elles, confirmé bug 30/07/2026).
    for doc_id in astreinte_coro_ids:
        for d_idx in range(5):
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
        # Weekend "combo" (confirmé utilisateur 30/07/2026) : 2 des 3
        # coronarographistes se répartissent astreinte ET garde sur le
        # weekend, désignés en entrée (~10 weekends/6 mois, hors mémoire du
        # solveur). L'ancre astreinte fait ven nuit + sam (matin/midi/nuit)
        # en astreinte, puis dimanche (matin/midi/nuit) en GARDE. L'ancre
        # garde fait l'inverse : samedi (matin/midi/nuit) en GARDE, dimanche
        # (matin/midi/nuit) en astreinte. Le 3e membre du pool WOM n'a ni
        # l'un ni l'autre ce weekend-là.
        combo_active = req.weekend_astreinte_combo
        if combo_active:
            # Weekend "combo" souple (confirmé utilisateur 30/07/2026) : les
            # ancres désignées sont des PRÉFÉRENCES, pas des forçages - si
            # l'ancre habituelle est absente (congé, formation), le solveur
            # réattribue automatiquement les rôles parmi les membres WOM
            # disponibles. Rôles modélisés par variable : is_astreinte_anchor
            # (ven nuit + sam ASTREINTE + dim GARDE) et is_garde_anchor (sam
            # GARDE + dim ASTREINTE), exactement 1 de chaque, jamais la même
            # personne aux deux rôles.
            is_astreinte_anchor: Dict[str, Any] = {}
            is_garde_anchor: Dict[str, Any] = {}
            for doc in wom_pool:
                fri_nuit_astreinte = x.get((doc, 4, "nuit", "ASTREINTE"))
                if fri_nuit_astreinte is None:
                    continue  # indisponible ce weekend (congé/formation) - ne peut jouer aucun rôle
                is_astreinte_anchor[doc] = model.NewBoolVar(f"combo_astreinte_anchor_{doc}")
                is_garde_anchor[doc] = model.NewBoolVar(f"combo_garde_anchor_{doc}")
                model.Add(is_astreinte_anchor[doc] + is_garde_anchor[doc] <= 1)

                model.Add(fri_nuit_astreinte == is_astreinte_anchor[doc])
                for slot in ("matin", "am", "nuit"):
                    v_sam_astr = x.get((doc, 5, slot, "ASTREINTE"))
                    if v_sam_astr is not None:
                        model.Add(v_sam_astr == is_astreinte_anchor[doc])
                    v_dim_garde = x.get((doc, 6, slot, "GARDE"))
                    if v_dim_garde is not None:
                        model.Add(v_dim_garde == is_astreinte_anchor[doc])
                    v_sam_garde = x.get((doc, 5, slot, "GARDE"))
                    if v_sam_garde is not None:
                        model.Add(v_sam_garde == is_garde_anchor[doc])
                    v_dim_astr = x.get((doc, 6, slot, "ASTREINTE"))
                    if v_dim_astr is not None:
                        model.Add(v_dim_astr == is_garde_anchor[doc])

                # Exclure l'ancre astreinte (Vendredi Nuit + WE) des astreintes de nuit de lundi (0) et mardi (1)
                # UNIQUE CONDITION : valable uniquement si M, O, W sont TOUS LES 3 PRÉSENTS pendant la semaine
                all_mow_present = all(not any(is_on_vacation(m_doc, d, req.vacations) for d in jours_semaine(week_start)) for m_doc in ["M", "O", "W"])
                if all_mow_present:
                    v_lundi_nuit = x.get((doc, 0, "nuit", "ASTREINTE"))
                    if v_lundi_nuit is not None:
                        model.Add(v_lundi_nuit == 0).OnlyEnforceIf(is_astreinte_anchor[doc])
                    
                    v_mardi_nuit = x.get((doc, 1, "nuit", "ASTREINTE"))
                    if v_mardi_nuit is not None:
                        model.Add(v_mardi_nuit == 0).OnlyEnforceIf(is_astreinte_anchor[doc])

            if is_astreinte_anchor:
                model.Add(sum(is_astreinte_anchor.values()) == 1)
            else:
                warnings.append("Weekend combo : aucun médecin W/O/M disponible pour le rôle astreinte.")
            if is_garde_anchor:
                model.Add(sum(is_garde_anchor.values()) == 1)
            else:
                warnings.append("Weekend combo : aucun médecin W/O/M disponible pour le rôle garde.")

            # Préférence souple pour les ancres désignées (bonus, pas forcé)
            if req.weekend_combo_astreinte_anchor in is_astreinte_anchor:
                combo_priority_bonus.append(20 * is_astreinte_anchor[req.weekend_combo_astreinte_anchor])
            if req.weekend_combo_garde_anchor in is_garde_anchor:
                combo_priority_bonus.append(20 * is_garde_anchor[req.weekend_combo_garde_anchor])

            # Éviter 2 rôles "garde" combo successifs pour le même médecin à
            # moins de 15 jours d'intervalle (confirmé utilisateur 30/07/2026)
            if req.last_combo_garde_doctor and req.last_combo_garde_date and req.last_combo_garde_doctor in is_garde_anchor:
                last_date = date.fromisoformat(req.last_combo_garde_date)
                gap_days = (days[5] - last_date).days
                if 0 <= gap_days < 15:
                    remaining_candidates = [d for d in is_garde_anchor if d != req.last_combo_garde_doctor]
                    if remaining_candidates:
                        model.Add(is_garde_anchor[req.last_combo_garde_doctor] == 0)
                    else:
                        warnings.append(
                            f"Weekend combo : {req.last_combo_garde_doctor} a fait garde combo il y a "
                            f"moins de 15 jours, mais aucun autre candidat disponible - repris quand même."
                        )
        else:
            # WOM sur les astreintes du weekend (Matin, AM, Nuit) - non-combo :
            # équité 6 mois déjà gérée par astreinte_g3_spread (objectif),
            # laisser le solveur choisir librement qui de M/O/W est le mieux
            # placé plutôt que de désigner quelqu'un en dur.
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

            # Astreinte nuit vendredi = astreinte nuit samedi, même médecin
            # (confirmé utilisateur 30/07/2026) - cas non-combo uniquement, le
            # combo gère déjà ça explicitement dans sa propre branche.
            for doc in wom_pool:
                v_ven = x.get((doc, 4, "nuit", "ASTREINTE"))
                v_sam = x.get((doc, 5, "nuit", "ASTREINTE"))
                if v_ven is not None and v_sam is not None:
                    model.Add(v_ven == v_sam)

    # --- 9bis. Règle absolue : jamais astreinte ET garde le même jour de
    # weekend pour un même médecin (confirmé utilisateur 30/07/2026) -
    # filet de sécurité indépendant du mécanisme (combo, non-combo) qui
    # aurait pu produire la situation. N'entre PAS en conflit avec une
    # saisie manuelle déjà en place pour ce jour précis (priorité absolue
    # des saisies manuelles, confirmée séparément) - dans ce cas, seul un
    # avertissement est émis plutôt qu'un blocage complet de la génération.
    astreinte_garde_rows = {"matin": "Astreintes ATL Matin", "am": "Astreintes ATL Midi", "nuit": "Astreintes ATL Nuit"}
    garde_rows_by_slot = {"matin": "Garde Matin", "am": "Garde Midi", "nuit": "Garde Nuit"}
    for doc_id in medecins_map:
        for d_idx in (5, 6):
            day_nm = DAY_NAMES_FR[d_idx]
            manual_astreinte = any(
                doc_id in (req.existing_schedule or {}).get(f"{r}||{day_nm}", [])
                for r in astreinte_garde_rows.values()
            )
            manual_garde = any(
                doc_id in (req.existing_schedule or {}).get(f"{r}||{day_nm}", [])
                for r in garde_rows_by_slot.values()
            )
            if manual_astreinte and manual_garde:
                warnings.append(
                    f"⚠️ {doc_id} a ASTREINTE et GARDE saisis manuellement le même jour "
                    f"({day_nm}) - normalement jamais cumulé, vérifiez cette saisie."
                )
                continue  # priorité à la saisie manuelle, pas de contrainte dure ici

            astreinte_vars = [
                v for (doc, d, sl, act), v in x.items()
                if doc == doc_id and d == d_idx and act == "ASTREINTE"
            ]
            garde_vars = [
                v for (doc, d, sl, act), v in x.items()
                if doc == doc_id and d == d_idx and act == "GARDE"
            ]
            if not astreinte_vars or not garde_vars:
                continue
            has_astreinte = model.NewBoolVar(f"has_astreinte_{doc_id}_{d_idx}")
            has_garde = model.NewBoolVar(f"has_garde_{doc_id}_{d_idx}")
            model.Add(sum(astreinte_vars) >= 1).OnlyEnforceIf(has_astreinte)
            model.Add(sum(astreinte_vars) == 0).OnlyEnforceIf(has_astreinte.Not())
            model.Add(sum(garde_vars) >= 1).OnlyEnforceIf(has_garde)
            model.Add(sum(garde_vars) == 0).OnlyEnforceIf(has_garde.Not())
            model.Add(has_astreinte + has_garde <= 1)

    # --- 10. Préservation des saisies manuelles ---
    if req.existing_schedule:
        for combined_key, doctors in req.existing_schedule.items():
            row_key, _, day_name = combined_key.partition("||")
            slot, activity = map_row_key_to_slot_activity(row_key)
            if slot is None or activity is None:
                continue
            day_idx = DAY_NAMES_FR.index(day_name)
            # Forcer les médecins présents à 1 - le solveur ne doit JAMAIS
            # modifier une saisie manuelle (confirmé utilisateur 30/07/2026).
            # Si une case saisie manuellement entre en conflit avec une règle
            # dure (ex: exclusion garde de nuit), le forçage échoue - il FAUT
            # le signaler explicitement plutôt que de laisser la case
            # redevenir silencieusement vide.
            for doc in doctors:
                var = x.get((doc, day_idx, slot, activity))
                if var is not None:
                    model.Add(var == 1)
                else:
                    warnings.append(
                        f"⚠️ Saisie manuelle non honorée : {doc} sur {row_key} ({day_name}) "
                        f"entre en conflit avec une règle du solveur - vérifiez cette case."
                    )
            # Forcer les autres à 0
            for doc in medecins_map:
                if doc not in doctors:
                    var = x.get((doc, day_idx, slot, activity))
                    if var is not None:
                        model.Add(var == 0)

    # Lun-Ven : ATL Matin/Midi = LA MÊME affectation que Coro matin/apm (pas
    # juste le même pool éligible) - confirmé DOC022 (28/07/2026). Concerne
    # M, O, W, FV (coronarographistes) ; CH n'a pas de var ici (structurel).
    for d_idx in range(5):
        for slot in ("matin", "am"):
            for doc in CORO_ALLOWED:
                v_astreinte = x.get((doc, d_idx, slot, "ASTREINTE"))
                v_coro = x.get((doc, d_idx, slot, "CORO"))
                if v_astreinte is not None and v_coro is not None:
                    model.Add(v_astreinte == v_coro)

    # --- 10bis. Couplages weekend (confirmé utilisateur 28/07/2026) ---
    # Tous ces couplages (par défaut, utile quand rien n'est saisi) N'ENTRENT
    # PLUS en conflit avec une saisie manuelle existante (confirmé bug
    # 30/07/2026, cf. règle 9bis) : si le médecin a déjà une saisie manuelle
    # sur AU MOINS une des 2 cases couplées ce jour-là, on ne force pas le
    # couplage pour lui - la saisie manuelle prime toujours.
    def _has_manual_entry(doc: str, row: str, day_nm: str) -> bool:
        return doc in (req.existing_schedule or {}).get(f"{row}||{day_nm}", [])

    ASTREINTE_ROW_BY_SLOT = {"matin": "Astreintes ATL Matin", "am": "Astreintes ATL Midi", "nuit": "Astreintes ATL Nuit"}
    GARDE_ROW_BY_SLOT = {"matin": "Garde Matin", "am": "Garde Midi", "nuit": "Garde Nuit"}

    # ATL Sam/Dim : Matin = Midi = Nuit (un seul médecin par jour)
    for d_idx in (5, 6):
        day_nm = DAY_NAMES_FR[d_idx]
        # Si N'IMPORTE QUEL médecin a une saisie manuelle sur l'un des 3
        # créneaux ASTREINTE ce jour-là, la journée n'est plus "propre" (un
        # seul médecin ne peut plus couvrir tout seul) - on désactive le
        # couplage automatique pour TOUT LE MONDE ce jour-là, pas seulement
        # pour le médecin concerné, sinon un autre WOM reste bloqué à devoir
        # couvrir toute la journée pour compenser un seul créneau déjà pris
        # (confirmé bug réel le 30/07/2026, cas W garde matin + astreinte nuit).
        day_has_any_manual_astreinte = any(
            _has_manual_entry(doc, r, day_nm)
            for doc in medecins_map
            for r in ASTREINTE_ROW_BY_SLOT.values()
        )
        if day_has_any_manual_astreinte:
            continue
        for slot_a, slot_b in (("matin", "am"), ("am", "nuit")):
            for doc in medecins_map:
                va = x.get((doc, d_idx, slot_a, "ASTREINTE"))
                vb = x.get((doc, d_idx, slot_b, "ASTREINTE"))
                if va is not None and vb is not None:
                    model.Add(va == vb)

    # Garde Samedi : Midi = Nuit (un seul médecin)
    samedi_has_any_manual_garde = any(
        _has_manual_entry(doc, r, "SAMEDI") for doc in medecins_map for r in GARDE_ROW_BY_SLOT.values()
    )
    if not samedi_has_any_manual_garde:
        for doc in medecins_map:
            va = x.get((doc, 5, "am", "GARDE"))
            vb = x.get((doc, 5, "nuit", "GARDE"))
            if va is not None and vb is not None:
                model.Add(va == vb)

    # Garde Dimanche : Matin = Midi = Nuit (un seul médecin)
    dimanche_has_any_manual_garde = any(
        _has_manual_entry(doc, r, "DIMANCHE") for doc in medecins_map for r in GARDE_ROW_BY_SLOT.values()
    )
    if not dimanche_has_any_manual_garde:
        for slot_a, slot_b in (("matin", "am"), ("am", "nuit")):
            for doc in medecins_map:
                va = x.get((doc, 6, slot_a, "GARDE"))
                vb = x.get((doc, 6, slot_b, "GARDE"))
                if va is not None and vb is not None:
                    model.Add(va == vb)

    # Garde Samedi Matin = celui qui a fait la garde de nuit vendredi
    # (Sam Midi/Nuit reste un choix séparé, déjà couplé entre eux ci-dessus)
    for doc in medecins_map:
        if _has_manual_entry(doc, "Garde Matin", "SAMEDI"):
            continue
        ven_nuit = x.get((doc, 4, "nuit", "GARDE"))
        sam_matin = x.get((doc, 5, "matin", "GARDE"))
        if ven_nuit is not None and sam_matin is not None:
            model.AddImplication(ven_nuit, sam_matin)

    # --- 11. Équité (objectif) ---
    # Restructuré le 28/07/2026 (confirmé utilisateur) : plus un seul pool
    # d'équité mélangeant tout, mais des groupes CLOISONNÉS pour éviter tout
    # double comptage (leçon tirée du bug ATL=Coro qui faisait exploser
    # l'équité générale). "Le principe d'équité ne doit pas créer de conflits."
    #
    # - Équité GARDE : les 11 médecins partageant la garde (A,S,B,H,G,P,M,O,W,U,Z)
    # - Groupe 3 (coronarographistes M,O,W) : ASTREINTE (nuit+weekend
    #   uniquement - le matin/midi semaine est déjà capté par CORO, voir
    #   couplage section 10) + CORO, comme deux métriques d'équité séparées.
    # - Groupe 1 (échographistes) et Groupe 2 (rythmologues) : voir note plus
    #   bas - pas encore de levier solveur actif pour eux à ce stade.

    WEIGHT_GARDE = 1
    GARDE_EQUITY_IDS = {"A", "S", "B", "H", "G", "P", "M", "O", "W", "U", "Z"} & set(medecins_map)

    garde_points: Dict[str, Any] = {}
    for doc in GARDE_EQUITY_IDS:
        m = medecins_map[doc]
        pct = m.poids_equite_pct if m.poids_equite_pct > 0 else 100
        norm = lambda w: round(w * 100 / pct)  # noqa: E731
        historical = m.points_garde * norm(WEIGHT_GARDE)
        this_week_terms = [
            norm(WEIGHT_GARDE) * var
            for (doc_id_x, d_idx, slot, activity), var in x.items()
            if doc_id_x == doc and activity == "GARDE"
        ]
        upper_bound = historical + 7 * norm(WEIGHT_GARDE) + 10
        total_var = model.NewIntVar(0, max(upper_bound, 1), f"garde_points_{doc}")
        model.Add(total_var == historical + sum(this_week_terms))
        garde_points[doc] = total_var

    if garde_points:
        max_points = model.NewIntVar(0, 100000, "max_points")
        min_points = model.NewIntVar(0, 100000, "min_points")
        for doc, var in garde_points.items():
            model.Add(max_points >= var)
            model.Add(min_points <= var)
    else:
        warnings.append("Équité garde : aucun médecin éligible trouvé")
        max_points = model.NewConstant(0)
        min_points = model.NewConstant(0)

    # --- Équité CORO (Groupe 3 : M, O, W uniquement - FV/CH non concernés) ---
    # Scope 6 MOIS glissants (pas cumulé à vie, pas mensuel) : m.points_coro
    # doit refléter cette fenêtre côté front.
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

    # --- Équité ASTREINTE ATL (Groupe 3 : M, O, W) - nuit + weekend
    # uniquement, le matin/midi semaine étant déjà capté par CORO ci-dessus
    # (ATL=Coro, même affectation - éviter le double comptage).
    WEIGHT_ASTREINTE_G3 = 1
    astreinte_g3_points: Dict[str, Any] = {}
    for doc in astreinte_coro_ids:
        m = medecins_map[doc]
        historical_astreinte = m.points_astreinte * WEIGHT_ASTREINTE_G3
        this_week_astreinte = [
            WEIGHT_ASTREINTE_G3 * var
            for (doc_id_x, d_idx, slot, activity), var in x.items()
            if doc_id_x == doc and activity == "ASTREINTE" and (d_idx in (5, 6) or slot == "nuit")
        ]
        upper_bound = historical_astreinte + 7 * WEIGHT_ASTREINTE_G3 + 5
        a_var = model.NewIntVar(0, max(upper_bound, 1), f"astreinte_g3_points_{doc}")
        model.Add(a_var == historical_astreinte + sum(this_week_astreinte))
        astreinte_g3_points[doc] = a_var

    # --- Équité élargie sur 6 mois pour tous les périmètres (Cs PSS/Tessée, ETT, EE) ---
    CS_ROW_KEYS = {"Matin - Cs PSS", "Matin - Cs Tessée", "Apm - Cs PSS", "Apm - Cs Tessée"}
    ETT_ROW_KEYS = {"Matin - ETT salle 1", "Matin - ETT salle 2", "Apm - ETT salle 1", "Apm - ETT salle 2", "Matin - ETT Tessée", "Apm - ETT Tessée"}
    EE_ROW_KEYS = {"Matin - EE1", "Apm - EE1", "Matin - EE2", "Apm - EE2", "Matin - EE", "Apm - EE"}
    STRESS_ROW_KEYS = {"Matin - Stress", "Apm - Stress"}

    ALL_CS_DOCTORS = (CS_PSS_ALLOWED | CS_TESSEE_ALLOWED) & set(medecins_map)
    ALL_ETT_DOCTORS = (ETT_ALLOWED | ETT_TESSEE_ALLOWED) & set(medecins_map)
    ALL_EE_DOCTORS = EE_ALLOWED & set(medecins_map)

    def _activity_equity_points(row_keys: Set[str], doctor_pool: Set[str], historical_getter, label: str) -> Dict[str, Any]:
        points: Dict[str, Any] = {}
        for doc in doctor_pool:
            m = medecins_map[doc]
            historical = historical_getter(m)
            this_week_terms = [
                var
                for (row_key, d_idx), day_vars in historical_vars.items()
                if row_key in row_keys
                for doc_id_x, (var, _freq) in day_vars.items()
                if doc_id_x == doc
            ]
            upper_bound = historical + 14
            var_total = model.NewIntVar(0, max(upper_bound, 1), f"eq_{label}_{doc}")
            model.Add(var_total == historical + sum(this_week_terms))
            points[doc] = var_total
        return points

    cs_points = _activity_equity_points(CS_ROW_KEYS, ALL_CS_DOCTORS, lambda m: m.points_cs, "cs")
    ett_points = _activity_equity_points(ETT_ROW_KEYS, ALL_ETT_DOCTORS, lambda m: m.points_ett, "ett")
    ee_points = _activity_equity_points(EE_ROW_KEYS, ALL_EE_DOCTORS, lambda m: getattr(m, "points_ee", 0), "ee")
    # Stress RETIRÉ de l'équité groupe 1 (confirmé utilisateur 29/07/2026) :
    # ce n'est pas une répartition égale mais un QUOTA fixe assumé inégal
    # (K=3, B/S/H/G/Z=1 chacun quand K est présent) - voir plus bas.

    # --- Quota Stress (K=3, B/S/H/G/Z=1 chacun si K présent) ---
    # PAS une équité de spread : un quota fixe assumé inégal. Pénalise l'écart
    # à la cible plutôt que de chercher l'égalité entre tous.
    STRESS_QUOTA = {"K": 3, "B": 1, "S": 1, "H": 1, "G": 1, "Z": 1}
    stress_quota_terms = []
    if "K" in medecins_map:
        for doc, target in STRESS_QUOTA.items():
            if doc not in medecins_map:
                continue
            count_vars = [
                var
                for (row_key, d_idx), day_vars in historical_vars.items()
                if row_key in STRESS_ROW_KEYS
                for doc_id_x, (var, _freq) in day_vars.items()
                if doc_id_x == doc
            ]
            if not count_vars:
                continue
            count_expr = sum(count_vars)
            dev = model.NewIntVar(0, 10, f"stress_dev_{doc}")
            model.Add(dev >= count_expr - target)
            model.Add(dev >= target - count_expr)
            stress_quota_terms.append(dev)
    stress_quota_penalty = sum(stress_quota_terms) if stress_quota_terms else 0

    # --- Bonus de fidélité historique (Cs/ETT/EE/hors site) ---
    # Récompense (dans l'objectif, donc soustrait puisqu'on minimise) le fait
    # d'assigner le médecin historiquement le plus fréquent sur ce créneau.
    # Poids volontairement modeste : ne doit pas dominer l'équité garde
    # (l'enjeu principal), juste départager entre plusieurs solutions
    # équivalentes en faveur de ce qui ressemble le plus au passé.
    WEIGHT_HISTORICAL_FIDELITY = 1
    historical_bonus_terms = [
        WEIGHT_HISTORICAL_FIDELITY * freq * var
        for day_vars in historical_vars.values()
        for (var, freq) in day_vars.values()
        if freq > 0
    ]
    historical_bonus = sum(historical_bonus_terms) if historical_bonus_terms else 0

    # --- Préférences souples ATL/Coro M/O/W (confirmé utilisateur 31/07/2026) ---
    jeudi_idx = DAY_NAMES_FR.index("JEUDI")

    # 1) Jeudi : celui de W/M qui NE fait PAS NCT fait la coro du matin (et
    # vice versa) - bonus pour le "complément", pas une obligation (O reste
    # candidat normal sur ce créneau par ailleurs).
    w_nct = x.get(("W", jeudi_idx, "nuit", "NCT"))
    m_nct = x.get(("M", jeudi_idx, "nuit", "NCT"))
    w_coro_matin_jeu = x.get(("W", jeudi_idx, "matin", "CORO"))
    m_coro_matin_jeu = x.get(("M", jeudi_idx, "matin", "CORO"))
    if w_nct is not None and m_coro_matin_jeu is not None:
        match1 = model.NewBoolVar("nct_coro_complement_w_m")
        model.AddBoolAnd([w_nct, m_coro_matin_jeu]).OnlyEnforceIf(match1)
        model.AddBoolOr([w_nct.Not(), m_coro_matin_jeu.Not()]).OnlyEnforceIf(match1.Not())
        hors_site_priority_bonus.append(20 * match1)
    if m_nct is not None and w_coro_matin_jeu is not None:
        match2 = model.NewBoolVar("nct_coro_complement_m_w")
        model.AddBoolAnd([m_nct, w_coro_matin_jeu]).OnlyEnforceIf(match2)
        model.AddBoolOr([m_nct.Not(), w_coro_matin_jeu.Not()]).OnlyEnforceIf(match2.Not())
        hors_site_priority_bonus.append(20 * match2)

    # 2) O souvent garde matin/midi/nuit le jeudi - option à privilégier dans
    # la mesure de l'équité globale, pas systématique (bonus léger).
    for slot in ("matin", "am", "nuit"):
        v_o_garde_jeu = x.get(("O", jeudi_idx, slot, "GARDE"))
        if v_o_garde_jeu is not None:
            hors_site_priority_bonus.append(5 * v_o_garde_jeu)

    # 3) Éviter de mettre W en coro le lendemain matin d'une astreinte de
    # nuit, quand M/O/W sont tous les 3 présents - préférence souple
    # (pénalité), pas une exclusion dure.
    for d_idx in range(6):
        w_astreinte_nuit = x.get(("W", d_idx, "nuit", "ASTREINTE"))
        w_coro_matin_next = x.get(("W", d_idx + 1, "matin", "CORO"))
        if w_astreinte_nuit is not None and w_coro_matin_next is not None:
            penalty = model.NewBoolVar(f"w_coro_after_astreinte_nuit_{d_idx}")
            model.AddBoolAnd([w_astreinte_nuit, w_coro_matin_next]).OnlyEnforceIf(penalty)
            model.AddBoolOr([w_astreinte_nuit.Not(), w_coro_matin_next.Not()]).OnlyEnforceIf(penalty.Not())
            hors_site_priority_bonus.append(-15 * penalty)

    hors_site_bonus = sum(hors_site_priority_bonus) if hors_site_priority_bonus else 0
    reeduc_bonus = sum(reeduc_priority_bonus) if reeduc_priority_bonus else 0

    # Préférence douce (pas une obligation) : P préfère la garde de nuit le
    # mercredi (confirmé utilisateur 29/07/2026) - petit bonus, n'empêche pas
    # P de faire garde nuit un autre jour si l'équité l'exige davantage.
    p_mercredi_var = x.get(("P", 2, "nuit", "GARDE"))
    p_wednesday_bonus = 3 * p_mercredi_var if p_mercredi_var is not None else 0
    entrees_pss_bonus = sum(entrees_pss_fill_bonus) if entrees_pss_fill_bonus else 0

    def _spread(points: Dict[str, Any], name: str):
        if not points:
            return 0
        pmax = model.NewIntVar(0, 100000, f"{name}_max")
        pmin = model.NewIntVar(0, 100000, f"{name}_min")
        for var in points.values():
            model.Add(pmax >= var)
            model.Add(pmin <= var)
        return pmax - pmin

    coro_spread = _spread(coro_points, "coro")
    astreinte_g3_spread = _spread(astreinte_g3_points, "astreinte_g3")
    cs_spread = _spread(cs_points, "cs")
    ett_spread = _spread(ett_points, "ett")
    ee_spread = _spread(ee_points, "ee")

    # Cible de 3 vacations semaine par médecin pour M, W, O quand tous les 3 sont présents
    mwo_target_terms = []
    mwo_present = [doc for doc in ["M", "W", "O"] if doc in medecins_map and not all(is_on_vacation(doc, days[d], req.vacations) for d in range(5))]
    if len(mwo_present) == 3:
        for doc in ["M", "W", "O"]:
            weekday_coro_vars = [
                var for (doc_id_x, d_idx, slot, activity), var in x.items()
                if doc_id_x == doc and activity == "CORO" and d_idx < 5
            ]
            if weekday_coro_vars:
                count_expr = sum(weekday_coro_vars)
                dev = model.NewIntVar(0, 5, f"mwo_weekday_coro_dev_{doc}")
                model.Add(dev >= count_expr - 3)
                model.Add(dev >= 3 - count_expr)
                mwo_target_terms.append(dev * 10)
    mwo_target_penalty = sum(mwo_target_terms) if mwo_target_terms else 0

    # --- Continuité des gardes (24h / même médecin) & Priorité la veille des 1/2 off AM ---
    garde_continuity_bonuses = []
    garde_eve_off_bonuses = []

    for d_idx, day_nm in enumerate(DAY_NAMES_FR):
        next_d_idx = d_idx + 1
        next_day_nm = DAY_NAMES_FR[next_d_idx] if next_d_idx < 7 else None

        # Médecins ayant leur 1/2 journée off l'après-midi du lendemain
        off_next_apm_docs = half_days_off.get((next_day_nm, "am"), set()) if next_day_nm else set()

        for doc_id in GARDE_EQUITY_IDS:
            v_matin = x.get((doc_id, d_idx, "matin", "GARDE"))
            v_am = x.get((doc_id, d_idx, "am", "GARDE"))
            v_nuit = x.get((doc_id, d_idx, "nuit", "GARDE"))

            # 1) Priorité pour les médecins dont la 1/2 journée off AM tombe le lendemain
            if doc_id in off_next_apm_docs and v_nuit is not None:
                garde_eve_off_bonuses.append(15 * v_nuit)

            # 2) Continuité 3 créneaux (24h complète : Matin + AM + Nuit)
            if v_matin is not None and v_am is not None and v_nuit is not None:
                is_24h = model.NewBoolVar(f"is_garde_24h_{doc_id}_{d_idx}")
                model.AddMinEquality(is_24h, [v_matin, v_am, v_nuit])
                garde_continuity_bonuses.append(30 * is_24h)

            # 3) Continuité 2 créneaux (si 1 créneau est pris par une tâche fixe)
            if v_am is not None and v_nuit is not None:
                is_am_nuit = model.NewBoolVar(f"is_garde_am_nuit_{doc_id}_{d_idx}")
                model.AddMinEquality(is_am_nuit, [v_am, v_nuit])
                garde_continuity_bonuses.append(10 * is_am_nuit)

            if v_matin is not None and v_am is not None:
                is_matin_am = model.NewBoolVar(f"is_garde_matin_am_{doc_id}_{d_idx}")
                model.AddMinEquality(is_matin_am, [v_matin, v_am])
                garde_continuity_bonuses.append(8 * is_matin_am)

    garde_continuity_bonus = sum(garde_continuity_bonuses) if garde_continuity_bonuses else 0
    garde_eve_off_bonus = sum(garde_eve_off_bonuses) if garde_eve_off_bonuses else 0

    # Un seul model.Minimize() possible avec CP-SAT : on combine l'équité
    # GARDE (poids fort, l'enjeu principal, 11 médecins), l'équité CORO et
    # ASTREINTE du groupe 3, l'équité Cs/ETT/EE pour l'ensemble des périmètres,
    # la continuité des gardes (24h / même médecin), la priorité la veille des 1/2 off AM,
    # le quota fixe Stress, le bonus de fidélité historique et les préférences hors site.
    combo_bonus = sum(combo_priority_bonus) if combo_priority_bonus else 0
    model.Minimize(
        (max_points - min_points) + coro_spread + astreinte_g3_spread
        + cs_spread + ett_spread + ee_spread + mwo_target_penalty + stress_quota_penalty + astreinte_nuit_coro_matin_penalty
        - historical_bonus - hors_site_bonus - reeduc_bonus - entrees_pss_bonus - p_wednesday_bonus - doublon_bonus - combo_bonus
        - garde_continuity_bonus - garde_eve_off_bonus
    )


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

        # --- Doublon Cs retenu : duplique l'assignation (le front interprète
        # 2 occurrences du même médecin dans la case comme un doublon "²") ---
        for doc_id, d_idx, slot, row_key, doublon_var in doublon_output_pairs:
            if solver.Value(doublon_var) == 1:
                clean_activity = row_key.split(" - ", 1)[1] if " - " in row_key else row_key
                assignments.append(Assignment(
                    date=days[d_idx].isoformat(),
                    day_name=DAY_NAMES_FR[d_idx],
                    slot=slot,
                    activity=clean_activity,
                    doctor=doc_id,
                    note="assigné par le solveur (doublon)"
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
